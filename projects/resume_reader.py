from pydantic import BaseModel, Field
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
import json
import os
import streamlit as st
from pathlib import Path

model = ChatOpenAI(model="gpt-4o", temperature=0.0)


class Experience(BaseModel):
    start_date: Optional[str] = Field(
        description="The start date of the experience, formatted as YYYY-MM-DD.")
    end_date: Optional[str] = Field(
        description="The end date of the experience, formatted as YYYY-MM-DD, or 'Present' if ongoing.")
    description: Optional[str] = Field(
        description="A brief description of the experience or role.")


class Study(Experience):
    degree: Optional[str] = Field(
        description="The degree obtained, such as Bachelor's, Master's, etc.")
    university: Optional[str] = Field(
        description="The name of the university or institution attended.")
    grade: Optional[str] = Field(
        description="The grade or GPA achieved during the study.")


class WorkExperience(Experience):
    company: str = Field(
        description="The name of the company where the work experience was gained.")
    job_title: str = Field(
        description="The title of the job held during the work experience.")


class Resume(BaseModel):
    first_name: str = Field(description="The first name of the individual.")
    last_name: str = Field(description="The last name of the individual.")
    github_url: Optional[str] = Field(
        description="The URL to the individual's Github profile.")
    linkedin_url: Optional[str] = Field(
        description="The URL to the individual's LinkedIn profile.")
    email_address: Optional[str] = Field(
        description="The individual's email address.")
    nationality: Optional[str] = Field(
        description="The nationality of the individual.")
    skill: Optional[str] = Field(
        description="A key skill or competency of the individual.")
    study: Optional[Study] = Field(
        description="Details of the individual's educational background.")
    work_experience: Optional[WorkExperience] = Field(
        description="Details of the individual's work experience.")
    hobby: Optional[str] = Field(
        description="A hobby or interest of the individual.")


prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

extraction_functions = [convert_to_openai_function(Resume)]
extraction_model = model.bind(
    functions=extraction_functions, function_call={"name": "Resume"})

extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()

uploaded_files = st.file_uploader(
    "Choose a txt file", type="pdf", accept_multiple_files=True)

output_file = "./data/extracted_resume.json"

if not os.path.exists(output_file):
    if uploaded_files:
        extracted_data = []

        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, mode='wb') as w:
                w.write(uploaded_file.getvalue())

            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(uploaded_file.name)
                pages = loader.load_and_split()

            # Clean up (delete) the file after processing
            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)
                print(f"File {uploaded_file.name} has bee deleted")

            response = extraction_chain.invoke({"input": pages[0]})

            # print(response)

            # Append the extracted information to a JSON file
            output_file = "data/extracted_resume.json"

            # Check if the JSON file exists and load its content if it does
            try:
                with open(output_file, 'r') as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = []

            # Append the new data to the existing content
            existing_data.append(response)

            # Write the updated content back to the JSON file
            with open(output_file, 'w') as file:
                json.dump(existing_data, file, indent=4)

            print(f"Resume data appended to {output_file}")


# Check if the output file exists
if os.path.exists(output_file):
    # Create a form
    with st.form("form"):
        # Input for the user's question
        question = st.text_input("Input your question about the documents")

        # Submit button
        submitted = st.form_submit_button("Submit")

        # Only proceed if the form is submitted
        if submitted:
            # Load the JSON data from the output file
            data = json.loads(Path(output_file).read_text())
            print(data)

            # Create the prompt using the ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the question based on the following context:"),
                ("human", "Question: {question}\nJSON Context: {context}")
            ])

            # Chain the prompt with the model
            chain = prompt | model

            # Invoke the chain with the question and context
            result = chain.invoke({"question": question, "context": data})

            # Display the result
            st.write(result.content)
