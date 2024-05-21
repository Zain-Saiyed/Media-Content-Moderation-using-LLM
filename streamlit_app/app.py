import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from openai import OpenAIError

from PIL import Image

import json
import os
import re

avatar_icon = {"user": Image.open('streamlit_app/assets/chat1.jpg'), "assistant": Image.open('streamlit_app/assets/1.jpg')}

movie_file_names = os.listdir('streamlit_app/subtitle')
movie_names = [" ".join(x.replace(".srt", "").split(".")).strip() for x in movie_file_names]

# main function to get subtitle corpus
def get_subtitle_text_corpus(subtitle_file):
    file_path = f"streamlit_app/subtitle/{subtitle_file}"
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as file:
        srt_content = file.read()

    # Clean the subtitle corpus
    srt_content = clean_srt_content(srt_content)

    return srt_content

def clean_srt_content(content):
  # Split the content into individual subtitle blocks
  subtitle_blocks = content.strip().split('\n\n')

  # Initialize a list to store cleaned subtitle blocks
  cleaned_blocks = []

  # Iterate through each subtitle block
  for block in subtitle_blocks:
    block = block.replace("<i>", "").replace("</i>", "")
    # Split the block into lines
    lines = block.strip().split('\n')
    # print("lines: ",lines)
    # Check if the block contains empty lines after timing information
    if len(lines) < 3 and not any(line.strip() for line in lines[2:]):
      # If the block contains empty lines, skip it
      continue
    else:
      if "[♪♪♪]" in block or "[♪♪♪♪]" in block or  "subtitles by" in block.lower():
        continue
      
      # Keep time information in subtitle
      # cleaned_text = lines[1]+" : "+' '.join(lines[2:])
      # Remove time information in subtitle
      cleaned_text = ' '.join(lines[2:])
      # print(cleaned_text)

      # If the block does not contain empty lines, add it to the cleaned list
      cleaned_blocks.append(cleaned_text)

  # Join the cleaned blocks to form the cleaned content
  cleaned_content = ' '.join(cleaned_blocks)

  return cleaned_content

# Split text into fixed size smaller chunks 
def get_text_chunks(complete_text_corpus):
    # text_splitter = CharacterTextSplitter(
    #     separator="\n", 
    #     chunk_size=1500, 
    #     chunk_overlap=800, 
    #     length_function=len
    # )
    # text_chunks = text_splitter.split_text(complete_text_corpus)
    # return text_chunks

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', complete_text_corpus)
    print(sentences)
    # Assigning ID to each sentence
    sentence_ids = [{"id": f"{i+1}", "text": sentence.strip()} for i, sentence in enumerate(sentences) if sentence.strip()]

    texts = [sentence['text'] for sentence in sentence_ids]

    return texts


# Create vector store with complete_text_corpus chunks and embeddings
def get_vectorstore(embeddings, text_chunks):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_response(chain_model, vector_store, user_query):
    # List me all the subtitles with the timing information in the given movie subtitles that are not suitable for audience less than the age of 12. Please explain the reasoning behind it.
    query = f"""
You are a professional movie content moderation expert.
List me all the subtitles with the timing information in the given movie subtitles that are not suitable for audience less than the age of 12. Please explain the reasoning behind it.

The subtitle should strictly only be listed in the generated output if any of the below criteria is satisfied:
Use of (any or all)
- Violence & Scariness, Murder or killing, Fights
- Sex, Romance & Nudity
- Offensive or Vulguar Language
- Religious Concerns
- Drinking, Drugs & Smoking

Along with the above categories also state the rating between 1 to 5.
List of categories above with rating between 1 and 5, and point wise all the subtitles that support that category be explicit.
"""
    print(query)
    documents = vector_store.similarity_search(query)
    response = chain_model.run(input_documents=documents, question=query)
    # print(response)
    return response

# Function to download chat history as JSON
def show_download_chat_history_button(selected_movie_name):
    chat_history_json = json.dumps(st.session_state["messages"], indent=4)

    # Display a download button
    st.download_button(
        label="Download Chat as JSON",
        data=chat_history_json,
        file_name=f'chat_history_{selected_movie_name}.json',
        mime='application/json'
    )

# MAIN Function
def main():
    # Set Header
    st.set_page_config(page_title="Chat with PDF :scroll:", page_icon=":scroll:")

    # set session variables
    if "selected_movie" not in st.session_state:
        st.session_state["selected_movie"] = None
        st.session_state["chat_enabled"] = False

    if "model_status" not in st.session_state:
        st.session_state["model_status"] = False
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = None
    
    if "movie_subtitle_selection" not in st.session_state:
        st.session_state["movie_subtitle_selection"] = False

    if not st.session_state["model_status"] and st.session_state["OPENAI_API_KEY"]:
        st.session_state["embeddings"]  = OpenAIEmbeddings(model="text-embedding-ada-002")
        st.session_state["chain_model"] = load_qa_chain(OpenAI(model="gpt-3.5-turbo-instruct",max_tokens=2120, temperature=0.7))
        st.session_state["model_status"] = True

    # Set Header
    st.header("Content Moderator AI - agent :books: ")
    st.text("Please choose a subtitle file from the sidebar to begin chatting.")

    # Side Bar:
    with st.sidebar:
        OPENAI_API_KEY = st.text_input('OPENAI API Key')
        search_term = st.text_input('Search for a movie')
        filtered_movies = [movie for movie in ([""] + movie_names) if search_term.lower() in movie.lower()]
        selected_movie = st.selectbox('Select a movie', filtered_movies)
        if selected_movie:
            st.write(f'You selected: {selected_movie}')
            st.session_state["movie_subtitle_selection"] = True

        st.session_state["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        button_click = st.button("Begin Chat")


    # Display error banner when:
    ### API Key is missing and starting chat
    if button_click and st.session_state["OPENAI_API_KEY"] == None:
        st.sidebar.error("Please Enter the API Key to start chat.")
        st.toast("Please Enter the :red[API Key] to start chat.")
    ### Movie selection is missing
    elif button_click and selected_movie == "":
        st.sidebar.error("Please select a movie to continue.")
        st.toast("Please select a :red[movie] to moderate content.")
    ### Start Chat
    elif button_click or st.session_state["chat_enabled"]:
        st.write(st.session_state["OPENAI_API_KEY"])
        # If selected movie is changed start new chat history
        if st.session_state["selected_movie"] != selected_movie:
            st.session_state["messages"] = []
        
        st.text("")
        st.session_state["chat_enabled"] = True
        
        # Display button to downlaod chat history
        show_download_chat_history_button(selected_movie)
        
        subtitle_file = ".".join(selected_movie.split(" "))+".srt"
        raw_text = get_subtitle_text_corpus(subtitle_file)

        # get the text chunks
        text_chunks = get_text_chunks(raw_text)
        print(text_chunks)
        
        try:

            # create vector store
            vectorstore = get_vectorstore(st.session_state["embeddings"], text_chunks)
            
            # Display message chat history
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"], avatar=avatar_icon[message["role"]]):
                    st.markdown(message["content"])

            # Get user's question
            if user_question := st.chat_input("What would you like to know about this movie?"):

                st.session_state["messages"].append({"role":"user", "content":user_question})

                response = get_response(st.session_state["chain_model"], vectorstore, user_question)
                print(response)
                st.session_state["messages"].append({"role":"assistant", "content":response})

                st.chat_message("user", avatar=avatar_icon["user"]).markdown(user_question)

                st.session_state["messages"].append({"role":"user", "content":user_question})

                with st.chat_message("assistant", avatar=avatar_icon["assistant"]):
                    st.markdown(f"Echo: {response}")

                st.session_state["messages"].append({"role":"assistant", "content":f"Echo: {response}"})
        except OpenAIError as e:
            st.error("Error : "+str(e))



if __name__ == "__main__":
    main()            

# streamlit run app.py --server.headless true