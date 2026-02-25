"""Prompt templates for SuperAgent."""

from __future__ import annotations

# Template for plan updates
PLAN_UPDATE_TEMPLATE = """
Current plan:
{current_plan}

Please update the plan based on recent actions and findings.
"""

# Template for reasoning
REASONING_TEMPLATE = """
Please analyze the current situation and decide on the next steps.
Consider:
1. What has been done so far?
2. What information is missing?
3. What is the most efficient way to proceed?
"""

# Template for self-verification before task completion
VERIFICATION_PROMPT_TEMPLATE = """<system-reminder>
# Self-Verification Required - CRITICAL

You indicated the task might be complete. Before finishing, you MUST perform a thorough self-verification.

## Original Task (re-read carefully):
{instruction}

## Self-Verification Checklist:

### 0. CRITICAL CONSTRAINT EXTRACTION (DO THIS FIRST!)
Before anything else, extract ALL critical constraints from the task that are easy to miss:

**Step A: Scan for quantity/scope constraints:**
- Look for any words or phrases that limit or specify quantities, counts, or exclusivity
- Numbers specifying counts, limits, or specific values

**Step B: Scan for location/path constraints:**
- Exact file paths, directory names, or locations specified
- Whether files should be in specific directories only

**Step C: Scan for format/naming constraints:**
- Specific file extensions, naming conventions, or formats required
- Case sensitivity requirements

**Step D: Scan for behavioral constraints:**
- Specific commands that must work (copy them exactly from the task)
- Expected outputs or return values
- Version requirements or compatibility constraints

**Step E: List your extracted constraints:**
Write out each critical constraint you found, then VERIFY EACH ONE explicitly below.

### 1. Requirements Analysis
- Re-read the ENTIRE original task above word by word
- List EVERY requirement, constraint, and expected outcome mentioned
- Check if there are any implicit requirements you might have missed

### 2. Critical Constraint Verification (MANDATORY)
For EACH constraint extracted in Step 0, explicitly verify compliance:
- If task specifies an exact path → Verify file exists at THAT EXACT path
- If task specifies exact commands → Run those EXACT commands and verify they work
- If task has quantity limits → Count and verify the quantities match
- Document the verification result for EACH constraint

### 3. Work Verification  
- For EACH requirement identified, verify it was completed:
  - Run commands to check file contents, test outputs, or verify state
  - Do NOT assume something works - actually verify it
  - If you created code, run it to confirm it works
  - If you modified files, read them back to confirm changes are correct
  - For tasks involving layered or incremental data: Verify that the extracted data includes ALL changes and updates, not just base data. Check that:
    * Item counts match the expected total (base items + changes/additions)
    * Updated items have their current/modified values, not original values
    * New items added through updates are present in the output
    * The data represents the complete current state after all modifications are applied

### 4. Component & Functionality Testing
1. Identify ALL components, modules, or functions mentioned in the task
2. For EACH component, write and run a test command that actually CALLS/EXERCISES the functionality (not just imports)
3. Run any custom tests you created to validate edge cases
4. If ANY test fails, analyze the failure, fix your solution, and re-run the tests
5. DO NOT give up if tests fail - iterate until all tests pass
6. For tasks with layered or incremental data: Test that the process correctly merges all data sources:
   - Verify that all data sources are accessible and properly processed
   - Confirm that querying or extracting returns the complete dataset (base + all changes)
   - Check that the output contains data that reflects the complete current state, not just the base state

### 5. Edge Cases & Quality
- Are there any edge cases the task mentioned that you haven't handled?
- Did you follow any specific format/style requirements mentioned?
- Are there any errors, warnings, or issues in your implementation?

#### 5. Workspace Cleanup (GC)
Before completing, clean up any artifacts YOU created that are not part of the deliverable:
- Remove compiled/cache files: `rm -rf __pycache__ *.pyc .pytest_cache` (if you created them)
- Remove build outputs: `rm -rf dist build *.egg-info node_modules` (if you installed/built them)
- Remove temp files: `rm -f *.tmp *.log` (any scratch files you created for testing)
- Keep ONLY files required by the task - remove intermediate or debug artifacts
- NEVER remove pre-existing files - only clean up what you created during this task

#### 6. Final Decision
After completing the above verification:
- If EVERYTHING is verified and correct (including ALL critical constraints): Summarize what was done and confirm completion
- If ANY constraint is violated or ANYTHING is missing or broken: Fix it now using the appropriate tools
- If files are in wrong location: MOVE them to correct location
- Remove any temporary or intermediate files created during development that are not explicitly required by the task

## CRITICAL REMINDERS:
- You are running in HEADLESS mode - DO NOT ask questions to the user
- DO NOT ask for confirmation or clarification - make reasonable decisions
- If something is ambiguous, make the most reasonable choice and proceed
- If you find issues during verification, FIX THEM before completing
- Only complete if you have VERIFIED (not assumed) that everything works

Proceed with verification now.
</system-reminder>"""

# Template for confirmation round after first verification
VERIFICATION_CONFIRMATION_TEMPLATE = """<system-reminder>
# Confirm Previous Verification - Use It, Then Complete or Fix Gaps Only

You just completed a verification round. Below is your **previous verification result**. Use it.

## Original Task:
{instruction}

## Your Previous Verification Result:
```
{previous_verification_result}
```

## What To Do Now:

1. **Use the previous verification result** – Review it. Treat it as your verification summary.

2. **Confirm correctness** – Decide: Was the previous verification successful? Did it cover all requirements and constraints from the task?

3. **Choose ONE of:**
   - **If previous verification was successful** (all requirements verified, nothing missing): Confirm that the task is complete. Do NOT use any tools. Just state that you confirm the verification and the task is done.
   - **If there were missing verifications** (something was not checked, or a constraint was skipped): Use tools **ONLY** to perform those missing checks. Do not re-do verification you already did. Only address the gaps. After completing the missing verification(s), either confirm completion (no tools) or report what remains.

4. **Do not repeat work** – Do not re-verify what you already verified. Only run tools for verifications that were missing.

5. **If task incomplete** – When you determine the task is incomplete (missing verifications, unmet requirements, or unresolved issues), you MUST include the exact phrase **task incomplete** in your response.

6. **Complete when ready** – Once you have confirmed (or completed missing verification and confirmed), respond without tool calls to finalize completion.

Proceed: confirm using the previous result, or run only missing verifications, then complete.
</system-reminder>"""

# Template for shell command timeout errors
SHELL_TIMEOUT_TEMPLATE = """Command timed out after {timeout}s.

The command may still be running in the background. Consider:
1. Check if the process is still running: `ps aux | grep <process>`
2. Increase timeout if the operation legitimately needs more time
3. Check if the command is waiting for input (use -y flags, heredocs, etc.)
4. Break the command into smaller steps

Partial output before timeout:
{output}"""

# Template for shell command execution errors
SHELL_ERROR_TEMPLATE = """Command failed with exit code {exit_code}.

Error analysis:
1. What does this error message indicate?
2. Is this a syntax error, missing dependency, permission issue, or logic error?
3. What is the root cause?
4. How can you fix it?

Output:
{output}"""


# Template for tool invalid parameters guidance
TOOL_INVALID_GUIDANCE_TEMPLATE = """
---
[INVALID TOOL PARAMETERS] The tool '{tool_name}' was called with invalid or missing parameters.

{error_details}

Please return the correct tool call with all required parameters properly specified."""

# Template for tool failure guidance
TOOL_FAILURE_GUIDANCE_TEMPLATE = """
---
[TOOL CALL FAILED] The tool '{tool_name}' did not execute successfully.

Please analyze and fix:
1. Check the error message above - what does it indicate?
2. Verify all required parameters were provided correctly
3. If it's a file/path issue: check if the path exists and is correct
4. If it's a command issue: verify the command syntax is valid
5. If parameters are missing: review the tool's required parameters

Next action: Fix the issue and retry with the correct tool call, or use an alternative approach."""
