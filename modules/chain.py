# Prompt Template
from langchain_core.prompts import ChatPromptTemplate
from services.llm_service import llm_with_retry, output_parser
import xml.etree.ElementTree as ET
import regex as re
from utils.decorator_helpers import try_n_times

# Checks the xml string and fixes it incase it is not well formatted
def xml_check_and_fix(xml_text: str, llm = 'groq'):
    """
    xml_text: str XML string to be checked and fixed
    llm: llm model to use
    Returns:
        str: Fixed XML string
    """
    # Replace & with &amp;
    xml_text = xml_text.replace('&', '&amp;')

    def check_xml(xml_text):
        # Check the xml
        try:
            # Parse XML to check well-formedness
            ET.fromstring(xml_text)
            return True
        except ET.ParseError as e:
            return f"XML is not well-formed: {e}"

    def fix_xml(xml_text, error, llm = llm):
        # Fix the xml with llm chain
        xml_fix_prompt = """The following xml is not well-formed. Your task is to fix the issue and return a well-formed xml.

Here's the error I get when trying to parse the xml - {error}:

XML text to fix:
{xml_text}

Make sure to only output the fixed xml. Don't output anything else.
"""
        xml_fix_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", xml_fix_prompt),
            ]
        )

        # Define the chain
        xml_fix_chain = xml_fix_prompt_template | llm_with_retry | output_parser
        xml_text = xml_fix_chain.with_config(configurable={"llm": llm}).invoke(
            {
                "error": error,
                "xml_text": xml_text
            }
        )
        return xml_text


    # Call the two functions in a loop until the xml is fixed
    while True:
            # Wrap around root text
            xml_text = f"<root>{xml_text}</root>"
            error = check_xml(xml_text)
            if error is True:
                return xml_text
            else:
                print(f'Fixing xml: {error}')
                # Fix the xml
                xml_text = fix_xml(xml_text, error, llm)


def escape_single_braces(text: str) -> str:
    """
    Replaces only truly single '{' or '}' with double braces.
    Existing double/triple braces remain unchanged.
    
    Examples:
      A single { brace } -> A single {{ brace }}
      Double {{ braces }} -> (unchanged) -> {{ braces }}
      Triple {{{ braces }}} -> (unchanged) -> {{{ braces }}}
    """

    # Regex to find a left brace '{' that is NOT preceded or followed by another '{'
    SINGLE_LEFT_BRACE = re.compile(r'(?<!\{)\{(?!\{)')

    # Regex to find a right brace '}' that is NOT preceded or followed by another '}'
    SINGLE_RIGHT_BRACE = re.compile(r'(?<!\})\}(?!\})')
    
    # Replace single '{' with '{{'
    text = SINGLE_LEFT_BRACE.sub('{{', text)
    # Replace single '}' with '}}'
    text = SINGLE_RIGHT_BRACE.sub('}}', text)
    return text

class Chain:
    def __init__(self, llm='groq', tags=None, use_xml_checker=False):
        self.llm = llm
        self.messages_list = []
        self.tags = tags
        self.use_xml_checker = use_xml_checker
        self.chain_steps = [self.call_llm_with_retry, self.parse_output]  # Default steps
        self.structured_output = None  # For optional structured output

    def add_message(self, role, content):
        """
        Adds a message to the chain.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'ai').
            content (str): The content of the message.
        """
        self.messages_list.append((role, content))

    def add_messages(self, messages):
        """
        Adds a list of messages to the chain.
        Args:
            messages (list): A list of messages to be added to the chain.
            Each message should be a tuple in the format (role, content).
        """
        self.messages_list.extend(messages)

    def set_llm(self, llm):
        self.llm = llm

    def add_chain_step(self, step_function):
        """
        Adds a function or process to the chain sequence.

        Args:
            step_function (callable): A function that will be executed as part of the chain.
        """
        self.chain_steps.append(step_function)

    def set_chain_steps(self, steps_list):
        """
        Sets the entire chain sequence with a list of functions.

        Args:
            steps_list (list): A list of functions to be executed in the chain.
        """
        self.chain_steps = steps_list

    def _build_chain(self):
        if not self.messages_list:
            raise ValueError("Messages list must have at least one message before building the chain.")
        
        # 1) Create a new list of sanitized messages
        sanitized_messages = []
        for message in self.messages_list:
            if isinstance(message, tuple) and len(message) == 2:
                role, content = message
                # Escape only truly single braces
                content = escape_single_braces(content)
                sanitized_messages.append((role, content))
            else:
                # If your messages_list can contain strings or special objects,
                # handle them accordingly. E.g. if it's a string, just escape it:
                if isinstance(message, str):
                    message = escape_single_braces(message)
                sanitized_messages.append(message)
        
        # 2) Build the prompt template from the sanitized messages
        chain = ChatPromptTemplate.from_messages(sanitized_messages)
        
        # 3) Attach any chain steps
        for step in self.chain_steps:
            chain |= step

        # 4) Optionally attach text extraction step
        if self.tags:
            chain |= self.extract_text_in_tags

        return chain


    def extract_text_in_tags(self, text: str):
        """
        Attempt to extract the desired tags. If it fails, call a self-correction
        step that does not appear in the user-visible message history.
        """
        max_corrections = 2  # how many times we attempt auto-correction
        attempt = 0
        
        while attempt < max_corrections:
            try:
                # Try extracting
                return self._attempt_extract_text_in_tags(text)
            except Exception as e:
                print(f"[!] Extraction error: {e}")
                # Attempt self-correction
                text = self._self_correct_output(text, str(e))
                attempt += 1
        
        # If we still fail after max_corrections, raise the original exception
        raise Exception(f"Unable to extract text from tags after {max_corrections} attempts.")

    def _attempt_extract_text_in_tags(self, text: str):
        """
        A helper that actually performs the extraction (no hidden LLM calls).
        Raises Exception if it can't find a required tag.
        """
        print(text)  # or use a logger
        texts = {'text': text}
        for tag in self.tags:
            pattern = f"<{tag}>\s*(.*?)\s*</{tag}>"
            match_1 = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match_1:
                texts[tag] = match_1.group(1)
            else:
                # Raise exception if a tag is missing
                raise Exception(f"Unable to extract the text from <{tag}> ... </{tag}> tags.")
        return texts

    def _self_correct_output(self, text, error_message):
        """
        Calls the LLM behind the scenes to correct the output format, using a local
        copy of the current conversation plus a correction message. Does *not* store
        these steps in self.messages_list, so they do not appear in user-facing history.
        """

        # 1. Make a local copy of the existing conversation
        ephemeral_messages = list(self.messages_list)

        # 2. Append a hidden correction message explaining the error and
        #    reminding the LLM of the original format requirements
        correction_message = (
            "SYSTEM: Your previous output had an error while extracting tags.\n\n"
            f"Error details: {error_message}\n\n"
            "Please regenerate a corrected output that satisfies the previously specified format.\n"
            "Reply with the full output. Don't make any changes to the message content, only fix the format by properly following the format specified."
        )
        ephemeral_messages.append(("user", correction_message))

        # 3. Pass the ephemeral messages (including the new correction message) to the LLM.
        #    This call does NOT modify self.messages_list, so the user never sees it.
        corrected_output = output_parser.invoke(llm_with_retry(
            ephemeral_messages,
            structured_output=self.structured_output,
            llm_name=self.llm
        ))

        return corrected_output



    def call_llm_with_retry(self, inputs):
        """
        Call the LLM using llm_with_retry and optionally structured output.
        Args:
            inputs: Input data for the LLM.
        Returns:
            The result of the LLM call.
        """
        return llm_with_retry(
            inputs,
            structured_output = self.structured_output,
            llm_name = self.llm
        )


    def parse_output(self, output):
        """
        Parses the output of the LLM. Adapts to structured or plain string output.
        Args:
            output: The raw output from the LLM.
        Returns:
            Parsed output, either as structured data or string output parser.
        """
        if self.structured_output:
            # Assume the output is already validated and structured if structured_output is used
            return output
        else:
            return output_parser


    # Decorator to retry a function up to n times if it raises an exception.
    @try_n_times(3)
    def run(self, **inputs):
        """
        Runs the chain with the provided inputs.

        Args:
            **inputs: The input parameters required by the chain.

        Returns:
            The result of the chain execution.
        """
        chain = self._build_chain()

        # Invoke the chain to get the response from the LLM
        response = chain.with_config(configurable={"llm": self.llm}).invoke(inputs)

        # If XML checker is enabled, validate the response
        if self.use_xml_checker:
            if isinstance(response, dict):
                response = response['text']
            #response = self.xml_checker(response)
            response = xml_check_and_fix(response, llm = self.llm)

        # Automatically update the message list with the AI response
        self.add_message('ai', response['text'] if isinstance(response, dict) else response)

        return response
