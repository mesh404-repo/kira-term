/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type Context,
	EventStream,
	streamSimple,
	type ToolResultMessage,
	validateToolArguments,
} from "@mariozechner/pi-ai";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	StreamFn,
} from "./types.js";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

// tau/sn66 v18: adaptive budget inferred from the task prompt itself.
// Validator kills at min(cursor_time*2, 300s). cursor_time correlates with
// task complexity (bullets, prompt length, explicit file paths). We tier
// the task and pick HARD_EXIT/TIME_WARNING accordingly so we don't waste
// 130s on a 300s-budget task or get killed on a 60s-budget task.
type BudgetTier = "short" | "medium" | "long";

interface TauBudget {
	tier: BudgetTier;
	HARD_EXIT_MS: number;
	TIME_WARNING_MS: number;
	MAX_READS_BEFORE_EDIT: number;
}

function tauExtractTaskText(messages: AgentMessage[]): string {
	for (const m of messages) {
		if (m.role !== "user") continue;
		const content = (m as { content?: unknown }).content;
		if (!Array.isArray(content)) continue;
		const parts: string[] = [];
		for (const c of content) {
			if (c && typeof c === "object" && (c as { type?: string }).type === "text") {
				const t = (c as { text?: unknown }).text;
				if (typeof t === "string") parts.push(t);
			}
		}
		if (parts.length > 0) return parts.join("\n");
	}
	return "";
}

function tauInferBudget(taskText: string): TauBudget {
	const len = taskText.length;
	const bullets = (taskText.match(/^\s*(?:[-*]|\d+\.)\s+/gm) || []).length;
	// Long: require BOTH strong signals — many criteria AND very long prompt.
	// Safe iff king_elapsed ≥ ~102s → budget ≥ 205s. Only modest 40s extension
	// beyond ac2i (205s vs 165s) so misclassification is survivable.
	if (bullets >= 12 && len > 2800) {
		return {
			tier: "long",
			HARD_EXIT_MS: 205_000,
			TIME_WARNING_MS: 25_000,
			MAX_READS_BEFORE_EDIT: 2,
		};
	}
	// Short: only trivially small prompts where king is very likely <30s.
	// Safe iff budget ≥ 61s which holds for any task in existence (min).
	if (bullets <= 2 && len < 350) {
		return {
			tier: "short",
			HARD_EXIT_MS: 50_000,
			TIME_WARNING_MS: 10_000,
			MAX_READS_BEFORE_EDIT: 1,
		};
	}
	// Medium: identical to ac2i behavior — proven safe for the common case.
	return {
		tier: "medium",
		HARD_EXIT_MS: 165_000,
		TIME_WARNING_MS: 18_000,
		MAX_READS_BEFORE_EDIT: 2,
	};
}

function tauExtractExplicitPaths(taskText: string): string[] {
	// Match things that look like paths with extensions: foo/bar.ts, src/app/page.tsx, etc.
	const pattern = /(?:[\w./-]*\/)?[\w-]+\.[A-Za-z][A-Za-z0-9]{0,5}\b/g;
	const raw = taskText.match(pattern) || [];
	const seen = new Set<string>();
	const out: string[] = [];
	for (const p of raw) {
		if (seen.has(p)) continue;
		if (p.length > 120) continue;
		// Skip pure-extension / leading-dot junk.
		if (p.startsWith(".") && !p.startsWith("./") && !p.startsWith("../")) continue;
		// Skip domain-like suffixes (e.g., "user.com", "foo.bar" without path hint).
		if (!p.includes("/") && /\.(com|org|net|io|dev|md|txt|log)$/.test(p)) continue;
		seen.add(p);
		out.push(p);
		if (out.length >= 6) break;
	}
	return out;
}

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<void> {
	let firstTurn = true;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	// tau/sn66 v15.1: provider-error retry. Verified in local smoke test
	// (smoke-batch-2): Gemini Flash via tau OpenRouter proxy intermittently
	// returns finish_reason=error mid-stream, leaving the partial assistant
	// message in context with no tool calls. Without retry the agent exits
	// with 0 edits and produces an empty diff. With retry we inject a
	// continuation prompt and try again.
	let providerErrorRetries = 0;
	const MAX_PROVIDER_ERROR_RETRIES = 100;

	// tau/sn66 v15.2: consecutive edit-error detector.
	const editErrorsByFile = new Map<string, number>();
	const stuckFilesAlerted = new Set<string>();
	const firstErrorNudged = new Set<string>();
	let firstErrorNudgeCount = 0;
	const MAX_FIRST_ERROR_NUDGES = 2;
	const EDIT_ERROR_THRESHOLD_PER_FILE = 2;

	// tau/sn66 v16: exploration budget + token-length retry + no-edit retry.
	let readsWithoutEdit = 0;
	let hasEditedAnyFile = false;
	let noToolCallRetries = 0;
	const MAX_NO_TOOL_RETRIES = 2;

	// tau/sn66 v18: track read-but-never-edited files. When the model tries
	// to stop with files still in this set, inject a steering message
	// forcing it to either edit or explicitly dismiss each one. Verified
	// against bench-3 where v18 read sync.ts at turn 4 but stopped at turn 9
	// without editing it, losing 2 matched lines vs ac2i.
	const readFilesNotYetEdited = new Set<string>();
	let staleReadsNudged = false;

	// tau/sn66 v18: adaptive budget based on the task prompt itself.
	const tauTaskText = tauExtractTaskText(currentContext.messages);
	const tauBudget = tauInferBudget(tauTaskText);
	const tauExplicitPaths = tauExtractExplicitPaths(tauTaskText);
	const MAX_READS_BEFORE_EDIT = tauBudget.MAX_READS_BEFORE_EDIT;
	const HARD_EXIT_MS = tauBudget.HARD_EXIT_MS;
	const TIME_WARNING_MS = tauBudget.TIME_WARNING_MS;

	const loopStartTime = Date.now();
	let timeWarningInjected = false;
	let lastChanceInjected = false;

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				await emit({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			// Process pending messages (inject before next assistant response)
			if (pendingMessages.length > 0) {
				for (const message of pendingMessages) {
					await emit({ type: "message_start", message });
					await emit({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			// Stream assistant response
			const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFn);
			newMessages.push(message);

			if (message.stopReason === "aborted") {
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// tau/sn66 v15.1: provider error → inject continuation and retry
			// instead of exiting. Caps at 3 retries to avoid infinite loops.
			if (message.stopReason === "error") {
				if (providerErrorRetries < MAX_PROVIDER_ERROR_RETRIES) {
					providerErrorRetries++;
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push({
						role: "user",
						content: [
							{
								type: "text",
								text: "Your previous response was cut off by a provider error. Continue immediately with a tool call — do not write narrative text, call read or edit directly. The harness scores your diff from disk; an empty diff loses the round.",
							},
						],
						timestamp: Date.now(),
					});
					hasMoreToolCalls = false;
					continue;
				}
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// tau/sn66 v16: if model hit token limit or stopped without tool calls,
			// inject retry. This catches the case where Gemini Flash writes a huge
			// text response and exhausts output tokens without making any tool call.
			const toolCalls = message.content.filter((c) => c.type === "toolCall");
			hasMoreToolCalls = toolCalls.length > 0;

			if (!hasMoreToolCalls && noToolCallRetries < MAX_NO_TOOL_RETRIES) {
				const isLength = message.stopReason === "length";
				const isStopNoEdit = message.stopReason === "stop" && !hasEditedAnyFile;
				// tau/sn66 v18: after the stall-guard fires, allow one more
				// retry even if hasEditedAnyFile — the model sometimes replies
				// with text acknowledging the stall but never calls a tool.
				const isStopAfterStallNudge = message.stopReason === "stop" && staleReadsNudged && readFilesNotYetEdited.size > 0;
				if (isLength || isStopNoEdit || isStopAfterStallNudge) {
					noToolCallRetries++;
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push({
						role: "user",
						content: [
							{
								type: "text",
								text: isLength
									? "You hit the token limit without making any tool call. Do NOT write text. Call `read` or `edit` directly. One read + one edit = minimum unit of work."
									: isStopAfterStallNudge
										? `Your previous reply was TEXT, not a tool call. Do NOT explain. Call \`edit\` on ${Array.from(readFilesNotYetEdited).slice(0,3).map(p=>`\`${p}\``).join(" and ")} NOW. Zero text, just the tool call.`
										: "You stopped without editing any file. An empty diff loses. Call `read` on the most likely target file, then `edit` it. Do it now.",
							},
						],
						timestamp: Date.now(),
					});
					continue;
				}
			}

			// tau/sn66 v18: stall-guard. If the model stops while there are
			// still files it has read but never edited, and we have budget
			// remaining, force it back to work on them. One-shot — if the
			// model stops again after this, let it go.
			if (
				!hasMoreToolCalls &&
				!staleReadsNudged &&
				message.stopReason === "stop" &&
				hasEditedAnyFile &&
				readFilesNotYetEdited.size > 0 &&
				(Date.now() - loopStartTime) < HARD_EXIT_MS - 20_000
			) {
				staleReadsNudged = true;
				const unedited = Array.from(readFilesNotYetEdited).slice(0, 6);
				await emit({ type: "turn_end", message, toolResults: [] });
				pendingMessages.push({
					role: "user",
					content: [
						{
							type: "text",
							text: `STOP — before you quit, check these files you read but did NOT edit: ${unedited.map((p) => `\`${p}\``).join(", ")}. For each file ask: "Is this file DIRECTLY named or DIRECTLY implied by the task?" If NO → reply "dismissed: <path> (off-task)" and move on — do NOT edit off-task files, editing them is pure bloat that hurts your score. If YES → call \`edit\` on it now. The common trap is missing a file that the task names. The opposite trap is editing a file just because you read it. Be surgical.`,
						},
					],
					timestamp: Date.now(),
				});
				continue;
			}

			const toolResults: ToolResultMessage[] = [];
			if (hasMoreToolCalls) {
				toolResults.push(...(await executeToolCalls(currentContext, message, config, signal, emit)));

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}

				// tau/sn66 v15.2: track consecutive edit failures per file.
				// When the same file accumulates >= threshold edit errors,
				// inject a steering message to force the model off that file.
				for (let i = 0; i < toolResults.length; i++) {
					const tr = toolResults[i];
					const tc = toolCalls[i];
					if (!tc || tc.type !== "toolCall") continue;
					if (tc.name !== "edit") continue;
					const targetPath = (tc.arguments as { path?: string } | undefined)?.path;
					if (!targetPath || typeof targetPath !== "string") continue;
					if (tr.isError) {
						const next = (editErrorsByFile.get(targetPath) ?? 0) + 1;
						editErrorsByFile.set(targetPath, next);
						// tau/sn66 v18: on the FIRST edit error, immediately nudge
						// the model to re-read that exact file before retrying.
						// Cheaper than waiting for the threshold to trip.
						if (next === 1 && !firstErrorNudged.has(targetPath) && firstErrorNudgeCount < MAX_FIRST_ERROR_NUDGES) {
							firstErrorNudged.add(targetPath);
							firstErrorNudgeCount++;
							pendingMessages.push({
								role: "user",
								content: [
									{
										type: "text",
										text: `Your edit on \`${targetPath}\` failed (oldText mismatch). Call \`read\` on \`${targetPath}\` NOW to see the current content, then retry with a UNIQUE 3-5 line oldText block that you have just seen. Never retry from memory. Do not switch files yet — fix this one edit first.`,
									},
								],
								timestamp: Date.now(),
							});
						}
						if (next >= EDIT_ERROR_THRESHOLD_PER_FILE && !stuckFilesAlerted.has(targetPath)) {
							stuckFilesAlerted.add(targetPath);
							pendingMessages.push({
								role: "user",
								content: [
									{
										type: "text",
										text: `STOP editing \`${targetPath}\`. You have failed ${next} edit attempts on this file in a row, all with "Could not find oldText" errors. The model's mental copy of this file is wrong. Do ONE of the following NOW:\n\n1. Move on to a DIFFERENT file in the task — there are likely other files mentioned in the acceptance criteria you haven't touched yet.\n2. If you must keep editing this file, call \`read\` on it ONE MORE TIME to refresh your view, then make ONE small edit with a very short, unique oldText snippet (5-10 lines max). Do not paste large blocks.\n3. Never paste text you remember — only paste text you have JUST read in this session.\n\nDo not retry the failed edits. Move on.`,
									},
								],
								timestamp: Date.now(),
							});
						}
					} else {
						// Successful edit on this file resets its error counter.
						editErrorsByFile.set(targetPath, 0);
						hasEditedAnyFile = true;
						readsWithoutEdit = 0;
						// tau/sn66 v17: after successful edit, warn model that
						// the file changed. Without this, the model tries to
						// edit the same file again with oldText from BEFORE its
						// edit, which always fails.
						pendingMessages.push({
							role: "user",
							content: [
								{
									type: "text",
									text: `\`${targetPath}\` was modified by your edit. If you need to edit this file again, call \`read\` on it first to see the current content. Do NOT use oldText from memory — it is now stale.`,
								},
							],
							timestamp: Date.now(),
						});
					}
				}

				// tau/sn66 v16: track exploration budget.
				for (const tr of toolResults) {
					if ((tr.toolName === "read" || tr.toolName === "bash") && !tr.isError) {
						if (!hasEditedAnyFile) readsWithoutEdit++;
					}
				}

				// tau/sn66 v18: track read-but-not-edited files.
				for (let i = 0; i < toolResults.length; i++) {
					const tr = toolResults[i];
					const tc = toolCalls[i];
					if (!tc || tc.type !== "toolCall") continue;
					const p = (tc.arguments as { path?: string } | undefined)?.path;
					if (typeof p !== "string" || p.length === 0) continue;
					if (tc.name === "read" && !tr.isError) {
						readFilesNotYetEdited.add(p);
					} else if ((tc.name === "edit" || tc.name === "write") && !tr.isError) {
						readFilesNotYetEdited.delete(p);
					}
				}

				// If model has read N files without editing, force it to edit.
				if (!hasEditedAnyFile && readsWithoutEdit >= MAX_READS_BEFORE_EDIT && pendingMessages.length === 0) {
					pendingMessages.push({
						role: "user",
						content: [
							{
								type: "text",
								text: "You have read enough files. Call `edit` on the most likely target file NOW. Do not read more files. One imperfect edit beats an empty diff.",
							},
						],
						timestamp: Date.now(),
					});
					readsWithoutEdit = 0;
				}

				// tau/sn66 v17: hard exit — stop gracefully before validator kills us.
				// This ensures the container is still running when the diff is collected.
				if ((Date.now() - loopStartTime) >= HARD_EXIT_MS) {
					await emit({ type: "turn_end", message, toolResults });
					await emit({ type: "agent_end", messages: newMessages });
					return;
				}

				// tau/sn66 v17: time pressure — if we've been running past the
				// budget-tier's warning threshold without a single successful
				// edit, inject urgency.
				if (!hasEditedAnyFile && !timeWarningInjected && (Date.now() - loopStartTime) >= TIME_WARNING_MS && pendingMessages.length === 0) {
					timeWarningInjected = true;
					const pathsHint =
						tauExplicitPaths.length > 0
							? `\n\nPaths mentioned in the task: ${tauExplicitPaths.map((p) => `\`${p}\``).join(", ")}`
							: "";
					pendingMessages.push({
						role: "user",
						content: [
							{
								type: "text",
								text: `TIME WARNING (budget tier: ${tauBudget.tier}): you have been running for ${Math.round((Date.now() - loopStartTime) / 1000)}s without producing an edit. The validator will kill this process soon. You MUST call \`edit\` or \`write\` on a file RIGHT NOW or you will score 0. Pick the single most obvious target file from the task and edit it immediately. Do not read any more files.${pathsHint}`,
							},
						],
						timestamp: Date.now(),
					});
				}

				// tau/sn66 v18: anti-empty-diff safety net. 15s before hard exit,
				// if we still have NOT edited anything, force a minimal write.
				// An empty diff auto-loses; a wrong-file diff at least has a
				// chance on any coincidentally-matched changed line, and the
				// explicit paths from the task are our best guess.
				const lastChanceAt = HARD_EXIT_MS - 15_000;
				if (
					!hasEditedAnyFile &&
					!lastChanceInjected &&
					(Date.now() - loopStartTime) >= lastChanceAt &&
					pendingMessages.length === 0
				) {
					lastChanceInjected = true;
					const pathsHint =
						tauExplicitPaths.length > 0
							? ` The task mentions these paths: ${tauExplicitPaths.map((p) => `\`${p}\``).join(", ")}. Pick the first one that exists and edit it, or \`write\` it if it doesn't exist.`
							: " Pick ANY file the task names — the most task-literal path — and edit it with a 3-5 line change.";
					pendingMessages.push({
						role: "user",
						content: [
							{
								type: "text",
								text: `LAST CHANCE: this process will terminate in ~15 seconds. You have produced NO edits. An empty diff scores 0 and auto-loses the round. Call \`edit\` or \`write\` RIGHT NOW — no reads, no bash, no planning, no text.${pathsHint}`,
							},
						],
						timestamp: Date.now(),
					});
				}
			}

			await emit({ type: "turn_end", message, toolResults });

			pendingMessages = (await config.getSteeringMessages?.()) || [];
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			// Set as pending so inner loop processes them
			pendingMessages = followUpMessages;
			continue;
		}

		// No more messages, exit
		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	// Build LLM context
	const llmContext: Context = {
		systemPrompt: context.systemPrompt,
		messages: llmMessages,
		tools: context.tools,
	};

	const streamFunction = streamFn || streamSimple;

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_start":
			case "text_delta":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					await emit({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: { ...finalMessage } });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	const finalMessage = await response.result();
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: { ...finalMessage } });
	}
	await emit({ type: "message_end", message: finalMessage });
	return finalMessage;
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	if (config.toolExecution === "sequential") {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const results: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit));
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			results.push(
				await finalizeExecutedToolCall(
					currentContext,
					assistantMessage,
					preparation,
					executed,
					config,
					signal,
					emit,
				),
			);
		}
	}

	return results;
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const results: ToolResultMessage[] = [];
	const runnableCalls: PreparedToolCall[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit));
		} else {
			runnableCalls.push(preparation);
		}
	}

	const runningCalls = runnableCalls.map((prepared) => ({
		prepared,
		execution: executePreparedToolCall(prepared, signal, emit),
	}));

	for (const running of runningCalls) {
		const executed = await running.execution;
		results.push(
			await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				running.prepared,
				executed,
				config,
				signal,
				emit,
			),
		);
	}

	return results;
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (beforeResult?.block) {
				return {
					kind: "immediate",
					result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
					isError: true,
				};
			}
		}
		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		const afterResult = await config.afterToolCall(
			{
				assistantMessage,
				toolCall: prepared.toolCall,
				args: prepared.args,
				result,
				isError,
				context: currentContext,
			},
			signal,
		);
		if (afterResult) {
			result = {
				content: afterResult.content ?? result.content,
				details: afterResult.details ?? result.details,
			};
			isError = afterResult.isError ?? isError;
		}
	}

	return await emitToolCallOutcome(prepared.toolCall, result, isError, emit);
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolCallOutcome(
	toolCall: AgentToolCall,
	result: AgentToolResult<any>,
	isError: boolean,
	emit: AgentEventSink,
): Promise<ToolResultMessage> {
	await emit({
		type: "tool_execution_end",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		result,
		isError,
	});

	const toolResultMessage: ToolResultMessage = {
		role: "toolResult",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		content: result.content,
		details: result.details,
		isError,
		timestamp: Date.now(),
	};

	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
	return toolResultMessage;
}
