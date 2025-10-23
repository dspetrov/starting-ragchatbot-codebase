"""
Tests for AIGenerator class

These tests verify that:
- AIGenerator correctly calls Anthropic API
- Tool execution flow works properly
- Tool results are passed back to Claude correctly
- Response formatting works
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from vector_store import SearchResults


class TestAIGeneratorInitialization:
    """Test AIGenerator initialization"""

    def test_init_with_api_key_and_model(self):
        """Test initializing AIGenerator with API key and model"""
        api_key = "test_api_key"
        model = "claude-sonnet-4-20250514"

        generator = AIGenerator(api_key, model)

        assert generator.model == model
        assert generator.base_params['model'] == model
        assert generator.base_params['temperature'] == 0
        assert generator.base_params['max_tokens'] == 800

    def test_system_prompt_is_set(self):
        """Test that system prompt is properly defined"""
        assert AIGenerator.SYSTEM_PROMPT
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT


class TestAIGeneratorBasicResponse:
    """Test basic response generation without tools"""

    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test generating a response without tool usage"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Generate response
        result = generator.generate_response(query="What is 2+2?")

        # Verify
        assert result["response"] == "This is a response"
        assert result["rounds_used"] == 0  # No tool rounds for direct answer
        assert result["tools_used"] == []
        mock_client.messages.create.assert_called_once()

    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test generating response with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Generate response with history
        history = "User: Previous question\nAssistant: Previous answer"
        result = generator.generate_response(
            query="Follow-up question",
            conversation_history=history
        )

        # Verify result format
        assert result["response"] == "Response with history"
        assert result["rounds_used"] == 0
        assert result["tools_used"] == []

        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args
        assert history in call_args.kwargs['system']


class TestAIGeneratorToolCalling:
    """Test tool calling functionality"""

    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_provided(self, mock_anthropic_class):
        """Test that tools are passed to the API when provided"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Create tool definitions
        tools = [
            {
                "name": "search_course_content",
                "description": "Search for course content",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]

        # Generate response with tools
        result = generator.generate_response(
            query="What is MCP?",
            tools=tools
        )

        # Verify result format
        assert result["response"] == "Response"
        assert result["rounds_used"] == 0
        assert result["tools_used"] == []

        # Verify tools were passed to API
        call_args = mock_client.messages.create.call_args
        assert 'tools' in call_args.kwargs
        assert call_args.kwargs['tools'] == tools
        assert 'tool_choice' in call_args.kwargs

    @patch('anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic_class):
        """
        CRITICAL TEST: Verify that tool execution flow works correctly.
        This tests the complete flow: Claude requests tool -> tool executes -> results sent back
        """
        # Setup mock client
        mock_client = Mock()

        # First response: Claude wants to use a tool
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "What is MCP?"}
        mock_tool_response.content = [mock_tool_block]

        # Second response: Final answer after tool use
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="MCP stands for Model Context Protocol")]
        mock_final_response.stop_reason = "end_turn"

        # Configure mock to return different responses
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Create tool manager with mock search tool
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["MCP is a protocol for AI"],
            metadata=[{"course_title": "MCP Course"}],
            distances=[0.1]
        )

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Generate response with tool
        result = generator.generate_response(
            query="What is MCP?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify tool was executed and final response returned
        assert result["response"] == "MCP stands for Model Context Protocol"
        assert result["rounds_used"] == 1  # One tool round
        assert result["tools_used"] == ["search_course_content"]
        assert mock_client.messages.create.call_count == 2  # Initial call + follow-up

    @patch('anthropic.Anthropic')
    def test_tool_execution_calls_tool_manager(self, mock_anthropic_class):
        """Test that tool execution properly calls the tool manager"""
        # Setup mocks
        mock_client = Mock()

        # Tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test query", "course_name": "Test Course"}
        mock_tool_response.content = [mock_tool_block]

        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        mock_final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Generate response
        result = generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Verify result format
        assert result["response"] == "Final answer"
        assert result["rounds_used"] == 1
        assert result["tools_used"] == ["search_course_content"]

        # Verify tool manager was called with correct parameters
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query",
            course_name="Test Course"
        )


class TestAIGeneratorToolExecutionHandler:
    """Test the _handle_tool_execution method"""

    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_builds_messages_correctly(self, mock_anthropic_class):
        """Test that tool execution handler builds message history correctly"""
        mock_client = Mock()

        # Final response after tool use
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        mock_final_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_final_response

        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Create initial response with tool use
        mock_initial_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}
        mock_initial_response.content = [mock_tool_block]

        # Create base params
        base_params = {
            "messages": [{"role": "user", "content": "What is MCP?"}],
            "system": "You are a helpful assistant"
        }

        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"

        # Execute tool handling
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )

        # Verify final API call was made with correct message structure
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs['messages']

        # Should have: original user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'
        assert messages[2]['role'] == 'user'

    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_without_tools_in_final_call(self, mock_anthropic_class):
        """Test that final API call after tool execution does NOT include tools parameter"""
        mock_client = Mock()

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        mock_final_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_final_response

        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Create initial response
        mock_initial_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}
        mock_initial_response.content = [mock_tool_block]

        base_params = {
            "messages": [{"role": "user", "content": "What is MCP?"}],
            "system": "You are a helpful assistant"
        }

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Results"

        # Execute
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )

        # Verify final call does NOT have 'tools' parameter
        call_args = mock_client.messages.create.call_args
        assert 'tools' not in call_args.kwargs


class TestAIGeneratorIntegration:
    """Integration tests with real tool manager"""

    @patch('anthropic.Anthropic')
    def test_full_tool_calling_flow(self, mock_anthropic_class):
        """
        Integration test: Full flow from query to tool execution to final answer.
        This simulates what happens when a user asks a content-related question.
        """
        # Setup mock Claude responses
        mock_client = Mock()

        # Response 1: Claude decides to use search tool
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_use_1"
        tool_block.input = {
            "query": "What is MCP?",
            "course_name": "Introduction to MCP"
        }
        tool_use_response.content = [tool_block]

        # Response 2: Claude gives final answer after seeing search results
        final_response = Mock()
        final_response.content = [Mock(
            text="Based on the course materials, MCP stands for Model Context Protocol."
        )]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]
        mock_anthropic_class.return_value = mock_client

        # Setup AI generator
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Setup tool manager with mock vector store
        mock_vector_store = Mock()
        mock_vector_store._resolve_course_name.return_value = "Introduction to MCP"
        mock_vector_store.search.return_value = SearchResults(
            documents=["MCP stands for Model Context Protocol. It enables AI applications to..."],
            metadata=[{
                "course_title": "Introduction to MCP",
                "lesson_number": 0,
                "lesson_link": "https://example.com/lesson-0"
            }],
            distances=[0.05]
        )

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Execute query
        result = generator.generate_response(
            query="What is MCP in the Introduction to MCP course?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify results
        assert "Model Context Protocol" in result["response"]
        assert result["rounds_used"] == 1
        assert result["tools_used"] == ["search_course_content"]
        assert mock_client.messages.create.call_count == 2

        # Verify search was called
        mock_vector_store.search.assert_called_once()

        # Verify sources are tracked
        sources = tool_manager.get_last_sources()
        assert len(sources) > 0


class TestMultiRoundToolCalling:
    """Test multi-round sequential tool calling functionality"""

    @patch('anthropic.Anthropic')
    def test_two_sequential_tool_calls(self, mock_anthropic_class):
        """Test Claude can make 2 separate tool calls across 2 rounds"""
        mock_client = Mock()

        # Round 1: First tool use
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "MCP basics"}
        round1_response.content = [tool_block_1]

        # Round 2: Second tool use after first tool results
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"course_name": "MCP Course"}
        round2_response.content = [tool_block_2]

        # Final response after both tools
        final_response = Mock()
        # Create a simple object with text attribute instead of Mock
        class TextBlock:
            text = "Here's what I found about MCP"
        final_response.content = [TextBlock()]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool results"

        # Execute
        result = generator.generate_response(
            query="Tell me about MCP",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager
        )

        # Verify
        assert result["response"] == "Here's what I found about MCP"
        assert result["rounds_used"] == 2  # Two tool rounds
        assert result["tools_used"] == ["search_course_content", "get_course_outline"]
        assert mock_client.messages.create.call_count == 3  # 3 API calls total

    @patch('anthropic.Anthropic')
    def test_max_rounds_enforcement(self, mock_anthropic_class):
        """Test system stops after 2 rounds even if Claude wants more"""
        mock_client = Mock()

        # Round 1: tool use
        tool_response_1 = Mock()
        tool_response_1.stop_reason = "tool_use"
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "test"}
        tool_response_1.content = [tool_block_1]

        # Round 2: tool use
        tool_response_2 = Mock()
        tool_response_2.stop_reason = "tool_use"
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"query": "test2"}
        tool_response_2.content = [tool_block_2]

        # Final response after max rounds
        final_response = Mock()
        class TextBlock:
            text = "Final answer after max rounds"
        final_response.content = [TextBlock()]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_response_1, tool_response_2, final_response]
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Results"

        # Execute
        result = generator.generate_response(
            query="Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Verify max rounds enforced
        assert result["rounds_used"] == 2  # Stopped at max
        assert mock_client.messages.create.call_count == 3  # 2 tool calls + 1 final call
        assert len(result["tools_used"]) == 2
        assert result["response"] == "Final answer after max rounds"

    @patch('anthropic.Anthropic')
    def test_tool_execution_error_terminates(self, mock_anthropic_class):
        """Test that tool execution error prevents additional rounds"""
        mock_client = Mock()

        # First response wants to use tool
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_1"
        tool_block.input = {"query": "test"}
        tool_response.content = [tool_block]

        mock_client.messages.create.return_value = tool_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        # Tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

        # Execute
        result = generator.generate_response(
            query="Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Verify error terminated the loop
        assert result["rounds_used"] == 0  # No successful rounds
        assert mock_client.messages.create.call_count == 1  # Only initial call
        assert result["tools_used"] == ["search_course_content"]  # Tool name collected before execution

    @patch('anthropic.Anthropic')
    def test_message_history_preserved_across_rounds(self, mock_anthropic_class):
        """Test messages accumulate correctly through rounds"""
        mock_client = Mock()

        # Capture message snapshots at each call
        captured_messages = []

        def capture_and_respond(**kwargs):
            # Make a copy of messages to capture state at this point
            captured_messages.append([msg.copy() for msg in kwargs.get('messages', [])])

            # Return appropriate response based on call count
            if len(captured_messages) == 1:
                # Round 1 - return tool use
                round1_response = Mock()
                round1_response.stop_reason = "tool_use"
                tool_block_1 = Mock()
                tool_block_1.type = "tool_use"
                tool_block_1.name = "search_course_content"
                tool_block_1.id = "tool_1"
                tool_block_1.input = {"query": "test"}
                round1_response.content = [tool_block_1]
                return round1_response
            else:
                # Round 2 - return final answer
                round2_response = Mock()
                text_block = Mock()
                text_block.text = "Final answer"
                round2_response.content = [text_block]
                round2_response.stop_reason = "end_turn"
                return round2_response

        mock_client.messages.create.side_effect = capture_and_respond
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool results"

        # Execute
        result = generator.generate_response(
            query="Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Verify message history structure at each call
        # First call should have: [user_message]
        assert len(captured_messages[0]) == 1
        assert captured_messages[0][0]['role'] == 'user'

        # Second call should have: [user_message, assistant_tool_use, user_tool_results]
        assert len(captured_messages[1]) == 3
        assert captured_messages[1][0]['role'] == 'user'
        assert captured_messages[1][1]['role'] == 'assistant'
        assert captured_messages[1][2]['role'] == 'user'

    @patch('anthropic.Anthropic')
    def test_tools_parameter_present_in_all_rounds(self, mock_anthropic_class):
        """Test that tools parameter is passed to API in every round"""
        mock_client = Mock()

        # Round 1
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_1"
        tool_block.input = {"query": "test"}
        round1_response.content = [tool_block]

        # Round 2 (final)
        round2_response = Mock()
        round2_response.content = [Mock(text="Final answer")]
        round2_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [round1_response, round2_response]
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Results"

        tools_list = [{"name": "search_course_content"}]

        # Execute
        result = generator.generate_response(
            query="Test query",
            tools=tools_list,
            tool_manager=mock_tool_manager
        )

        # Verify tools present in ALL API calls
        for call in mock_client.messages.create.call_args_list:
            assert 'tools' in call.kwargs
            assert call.kwargs['tools'] == tools_list
            assert 'tool_choice' in call.kwargs

    @patch('anthropic.Anthropic')
    def test_conversation_history_preserved_in_multi_round(self, mock_anthropic_class):
        """Test conversation history is maintained across tool rounds"""
        mock_client = Mock()

        # Round 1
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_1"
        tool_block.input = {"query": "test"}
        round1_response.content = [tool_block]

        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Final answer")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [round1_response, final_response]
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        generator.client = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Results"

        conversation_history = "User: Previous question\nAssistant: Previous answer"

        # Execute
        result = generator.generate_response(
            query="Follow-up question",
            conversation_history=conversation_history,
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Verify conversation history in system prompt for all calls
        for call in mock_client.messages.create.call_args_list:
            assert conversation_history in call.kwargs['system']
