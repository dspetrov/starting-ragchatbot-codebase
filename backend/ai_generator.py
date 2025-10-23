from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and
educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: Search for specific course content and educational materials
2. **get_course_outline**: Retrieve complete course structure including course title,
   course link, instructor, and all lessons with their numbers and titles

Tool Usage Guidelines:
- Use **search_course_content** for questions about specific course content or
  detailed educational materials
- Use **get_course_outline** for questions about course structure, lesson listings,
  or course overview
- **Maximum two tool rounds per query** - You can use tools, analyze results,
  then use tools again if needed
- **Use tools efficiently** - Prefer comprehensive single searches over multiple
  narrow ones
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Multi-Round Tool Strategy:
- **Round 1**: If initial search yields insufficient results, you may search again
  with different parameters
- **Round 2**: Use for follow-up searches (e.g., different course, broader query,
  or outline retrieval)
- **Examples**:
  - Round 1: Search "MCP basics" → Round 2: Get course outline for structure
  - Round 1: Search specific lesson → Round 2: Search related lesson for comparison
  - Round 1: Search returns no results → Round 2: Try broader search or different course

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content first, then answer
- **Course outline/structure questions**: Use get_course_outline first, then answer
  with complete details
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or
   question-type analysis
 - Do not mention "based on the tool results" or similar phrases

Outline Response Format:
When using get_course_outline, always include in your response:
- Course title
- Course link
- Instructor name
- Complete list of lessons with:
  - Lesson number
  - Lesson title
  - Lesson link (if available)

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> Dict[str, Any]:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential tool calling rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Dictionary with response and metadata:
            {
                "response": str,           # Final answer text
                "rounds_used": int,        # Number of tool rounds executed
                "tools_used": List[str]    # Tool names in order of execution
            }
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize tracking variables
        messages = [{"role": "user", "content": query}]
        round_count = 0
        MAX_ROUNDS = 2  # Maximum sequential tool calling rounds
        all_tools_used = []
        latest_response = None

        # Prepare base API parameters (preserved across rounds)
        base_api_params = {**self.base_params, "system": system_content}

        # Add tools if available (tools persist across all rounds)
        if tools:
            base_api_params["tools"] = tools
            base_api_params["tool_choice"] = {"type": "auto"}

        # Multi-round tool calling loop
        while True:
            # Make API call with current messages
            api_params = {**base_api_params, "messages": messages}
            latest_response = self.client.messages.create(**api_params)

            # TERMINATION CONDITION 1: Claude finished without tools
            if latest_response.stop_reason != "tool_use":
                break

            # TERMINATION CONDITION 2: No tool_use blocks (safety check)
            tool_use_blocks = [
                block for block in latest_response.content if block.type == "tool_use"
            ]
            if not tool_use_blocks:
                break

            # TERMINATION CONDITION 3: No tool manager provided
            if not tool_manager:
                break

            # Execute tools for this round
            try:
                messages, tool_success, tool_names = self._execute_tools_for_round(
                    latest_response, messages, tool_manager
                )
                all_tools_used.extend(tool_names)

                # TERMINATION CONDITION 4: Tool execution failed
                if not tool_success:
                    break

            except Exception as e:
                # Log error and break loop to prevent infinite loops
                print(f"Tool execution error in round {round_count}: {e}")
                break

            # Increment round counter
            round_count += 1

            # TERMINATION CONDITION 5: Max rounds reached
            # After executing tools, check if we've hit max rounds
            # If so, make one final call to get Claude's response
            if round_count >= MAX_ROUNDS:
                api_params = {**base_api_params, "messages": messages}
                latest_response = self.client.messages.create(**api_params)
                break

        # Extract final response text
        response_text = ""
        if latest_response:
            for block in latest_response.content:
                if hasattr(block, "text"):
                    response_text = block.text
                    break

        # Return response with metadata
        return {
            "response": response_text if response_text else "Error: No response generated",
            "rounds_used": round_count,
            "tools_used": all_tools_used,
        }

    def _execute_tools_for_round(
        self, response, messages: List[Dict], tool_manager
    ) -> tuple[List[Dict], bool, List[str]]:
        """
        Execute all tool calls from Claude's response and update message history.

        Args:
            response: Anthropic API response containing tool_use blocks
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (updated_messages, success_flag, tool_names)
            - updated_messages: Messages list with assistant response and tool results added
            - success_flag: True if all tools executed successfully, False otherwise
            - tool_names: List of tool names that were executed
        """
        # Append assistant's tool use response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool_use blocks and collect results
        tool_results = []
        all_succeeded = True
        tool_names = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_names.append(content_block.name)
                try:
                    # Execute the tool
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    # Check if tool returned an error message
                    if tool_result and isinstance(tool_result, str) and "Error" in tool_result:
                        all_succeeded = False

                    # Add result to collection
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

                except Exception as e:
                    # Tool execution failed - record error
                    all_succeeded = False
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True,
                        }
                    )

        # Append tool results as user message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages, all_succeeded, tool_names

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        DEPRECATED: This method is kept for backward compatibility only.
        Use the new multi-round tool calling loop in generate_response() instead.

        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(content_block.name, **content_block.input)

                tool_results.append(
                    {"type": "tool_result", "tool_use_id": content_block.id, "content": tool_result}
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {**self.base_params, "messages": messages, "system": base_params["system"]}

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
