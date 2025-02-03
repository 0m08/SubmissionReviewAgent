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


class Chain:
    def __init__(self, llm='groq', tags=None, use_xml_checker=False):
        self.llm = llm
        self.messages_list = []
        self.tags = tags
        self.use_xml_checker = use_xml_checker
        self.chain_steps = [llm_with_retry, output_parser]  # Default steps

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

        chain = ChatPromptTemplate.from_messages(self.messages_list)

        for step in self.chain_steps:
            chain |= step

        # Conditionally add the extract_text_in_tags step if tags are provided
        if self.tags:
            chain |= self.extract_text_in_tags

        return chain

    def extract_text_in_tags(self, text: str):
        """
        Extract a list of texts within the specified XML tags.

        Args:
            text (str): The input text containing the XML tags.

        Returns:
            list: A list of texts extracted from the specified tags.
        """
        print(text)
        texts = {'text': text}  # Store the text within each tag as a list
        for tag in self.tags:
            pattern = f"<{tag}>\s*(.*?)\s*</{tag}>"
            match_1 = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match_1:
                texts[tag] = match_1.group(1)
            else:
                raise Exception(f"Unable to extract the text from the {tag} tags")
        return texts

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
