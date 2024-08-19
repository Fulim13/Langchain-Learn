# from langchain_community.document_loaders.image import UnstructuredImageLoader

# loader = UnstructuredImageLoader("./data/image11.png")

# data = loader.load()

# print(data[0])


from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate


loader = WebBaseLoader(
    ["https://github.com/Fulim13/PythonLearning/blob/main/primitive.py",
        "https://github.com/Fulim13/PythonLearning/blob/main/control.py"])


docs = loader.load()

# print(docs)

for doc in docs:

    # Strip all \n
    doc_str = doc.page_content.replace('\n\n', '')
    # print(doc_str)

    # Model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # Template
    template = """Convert this context to a python markdown note that can let people to study,
    only extract python related information,
    usually python related information stick together

    Context: {context}
    """

    # Prompt
    prompt = PromptTemplate(input_variables=["context"],
                            template=template)

    formatted_prompt = prompt.format(context=doc_str)

    response = llm.invoke(formatted_prompt)

    # Write the content to a text file
    with open("document_content.txt", "a") as file:
        file.write(response.content)

    print("Document content has been saved to document_content.txt")
