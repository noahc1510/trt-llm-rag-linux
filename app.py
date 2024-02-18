# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import time
import json
import logging
import gc
import torch
from pathlib import Path
from trt_llama_api import TrtLlmAPI
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from collections import defaultdict
from llama_index import ServiceContext
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import set_global_service_context
from faiss_vector_storage import FaissEmbeddingStorage
from ui.user_interface import MainInterface

app_config_file = 'config/app_config.json'
model_config_file = 'config/config.json'
preference_config_file = 'config/preferences.json'
data_source = 'directory'


def read_config(file_name):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except json.JSONDecodeError:
        print(f"There was an error decoding the JSON from the file {file_name}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def get_model_config(config, model_name=None):
    models = config["models"]["supported"]
    selected_model = next((model for model in models if model["name"] == model_name), models[0])
    return {
        "model_path": os.path.join(os.getcwd(), selected_model["metadata"]["model_path"]),
        "engine": selected_model["metadata"]["engine"],
        "tokenizer_path": os.path.join(os.getcwd(), selected_model["metadata"]["tokenizer_path"]),
        "max_new_tokens": selected_model["metadata"]["max_new_tokens"],
        "max_input_token": selected_model["metadata"]["max_input_token"],
        "temperature": selected_model["metadata"]["temperature"]
    }


def get_data_path(config):
    return os.path.join(os.getcwd(), config["dataset"]["path"])

# read the app specific config
app_config = read_config(app_config_file)
streaming = app_config["streaming"]
similarity_top_k = app_config["similarity_top_k"]
is_chat_engine = app_config["is_chat_engine"]
embedded_model = app_config["embedded_model"]
embedded_dimension = app_config["embedded_dimension"]

# read model specific config
selected_model_name = None
selected_data_directory = None
config = read_config(model_config_file)
if os.path.exists(preference_config_file):
    perf_config = read_config(preference_config_file)
    selected_model_name = perf_config.get('models', {}).get('selected')
    selected_data_directory = perf_config.get('dataset', {}).get('path')

if selected_model_name == None:
    selected_model_name = config["models"].get("selected")

model_config = get_model_config(config, selected_model_name)
trt_engine_path = model_config["model_path"]
trt_engine_name = model_config["engine"]
tokenizer_dir_path = model_config["tokenizer_path"]
data_dir = config["dataset"]["path"] if selected_data_directory == None else selected_data_directory

# create trt_llm engine object
llm = TrtLlmAPI(
    model_path=model_config["model_path"],
    engine_name=model_config["engine"],
    tokenizer_dir=model_config["tokenizer_path"],
    temperature=model_config["temperature"],
    max_new_tokens=model_config["max_new_tokens"],
    context_window=model_config["max_input_token"],
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False
)

# create embeddings model object
embed_model = HuggingFaceEmbeddings(model_name=embedded_model)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model,
                                               context_window=model_config["max_input_token"], chunk_size=512,
                                               chunk_overlap=200)
set_global_service_context(service_context)


def generate_inferance_engine(data, force_rewrite=False):
    """
       Initialize and return a FAISS-based inference engine.

       Args:
           data: The directory where the data for the inference engine is located.
           force_rewrite (bool): If True, force rewriting the index.

       Returns:
           The initialized inference engine.

       Raises:
           RuntimeError: If unable to generate the inference engine.
       """
    try:
        global engine
        faiss_storage = FaissEmbeddingStorage(data_dir=data,
                                              dimension=embedded_dimension)
        faiss_storage.initialize_index(force_rewrite=force_rewrite)
        engine = faiss_storage.get_engine(is_chat_engine=is_chat_engine, streaming=streaming,
                                          similarity_top_k=similarity_top_k)
    except Exception as e:
        raise RuntimeError(f"Unable to generate the inference engine: {e}")


# load the vectorstore index
generate_inferance_engine(data_dir)

# start utube code
from pytube import Playlist, YouTube, extract
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

formatter = TextFormatter()


def identify_url_type(url: str) -> str:
    """
    Identify whether the given URL is a YouTube video or a playlist.

    :param url: URL to be checked
    :return: String indicating the type of URL ('video', 'playlist', or 'unknown')
    """
    try:
        # Check if the URL is a YouTube playlist URL
        if extract.playlist_id(url):
            return 'playlist'
    except Exception:
        pass

    try:
        # Check if the URL is a YouTube video URL
        if extract.video_id(url):
            return 'video'
    except Exception:
        pass

    # If neither, return 'unknown'
    return 'unknown'


def extract_video_id(url):
    # Extract the video ID from the URL
    try:
        return url.split("v=")[1].split("&")[0]
    except IndexError:
        return None


def dict_to_xml(tag, d):
    elem = ET.Element(tag)
    for key, val in d.items():
        child = ET.SubElement(elem, key)
        child.text = str(val)
    return ET.tostring(elem, encoding='unicode')


def fetch_transcripts(url, num_of_videos_download=None):
    """
    Fetches and saves transcripts of YouTube videos or playlists as XML files.

    Args:
        url (str): The YouTube URL (either a video or a playlist).
        num_of_videos_download (int, optional): Maximum number of video transcripts to download from a playlist.
                                                If None, all videos in the playlist will be processed.

    Returns:
        bool: True if all transcripts are successfully downloaded and saved,
              False if an error occurs or if the URL is invalid.
    """
    url_type = identify_url_type(url)
    if url_type == 'video':
        video_ids = [extract_video_id(url)]
    elif url_type == 'playlist':
        playlist = Playlist(url)
        video_ids = [video.split('=')[-1] for video in playlist.video_urls]
        if num_of_videos_download is not None:
            video_ids = video_ids[:int(num_of_videos_download)]
    else:
        logging.error(f"Unknown URL type, URL is invalid: {url}")
        return False

    data_path = Path(os.getcwd()) / 'youtube_dataset'
    data_path.mkdir(exist_ok=True)
    success = True  # Flag to track if all transcripts are successfully downloaded
    for video_id in video_ids:
        try:
            yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            json_formatted = formatter.format_transcript(transcript).replace("\n", " ")
            video_details = {
                "title": yt.title,
                "channel": yt.author,
                "upload_date": yt.publish_date.strftime("%Y-%m-%d"),
                "transcript": json_formatted
            }
            xml_data = dict_to_xml("video", video_details)
            with open(data_path / f"{video_id}.xml", "w", encoding="utf-8") as xml_file:
                xml_file.write(xml_data)
        except Exception as e:
            logging.error(f"Error fetching XML for video {video_id}: {e}")
            success = False  # Set flag to False if any error occurs
    return success

# end utube code

# chat function to trigger inference
import xml.etree.ElementTree as ET

def call_llm_streamed(query):
    partial_response = ""
    response = llm.stream_complete(query)
    for token in response:
        partial_response += token.delta
        yield partial_response

def chatbot(query, chat_history, session_id):
    if data_source == "nodataset":
        yield llm.complete(query).text
        return

    if is_chat_engine:
        response = engine.chat(query)
    else:
        response = engine.query(query)

    # Aggregate scores by file
    file_scores = defaultdict(float)
    for node in response.source_nodes:
        metadata = node.metadata
        if 'filename' in metadata:
            file_name = metadata['filename']
            file_scores[file_name] += node.score

    # Find the file with the highest aggregated score
    highest_aggregated_score_file = None
    if file_scores:
        highest_aggregated_score_file = max(file_scores, key=file_scores.get)

    file_links = []
    seen_files = set()  # Set to track unique file names

    # Generate links for the file with the highest aggregated score
    if highest_aggregated_score_file:
        abs_path = Path(os.path.join(os.getcwd(), highest_aggregated_score_file.replace('\\', '/')))
        file_name = os.path.basename(abs_path)
        file_name_without_ext = abs_path.stem
        if file_name not in seen_files:  # Ensure the file hasn't already been processed
            if data_source == 'youtube':
                # Attempt to read and parse the XML file for YouTube data
                try:
                    tree = ET.parse(abs_path)
                    root = tree.getroot()
                    title = root.find('title').text
                    channel = root.find('channel').text
                    upload_date = root.find('upload_date').text
                    youtube_info = f"<b>Title:</b> {title} <br> <b>Channel:</b> {channel} <br> <b>Upload Date:</b> {upload_date}"
                except ET.ParseError:
                    youtube_info = "Failed to parse XML file"

                file_link = f"<a href='https://www.youtube.com/watch?v={file_name_without_ext}'><img src='https://img.youtube.com/vi/{file_name_without_ext}/default.jpg'/></a>{youtube_info}<br>"
            elif data_source == 'directory':
                file_link = f'<a href="file:////{abs_path}">{file_name}</a>'
            else:
                exit("Wrong data_source type")
            file_links.append(file_link)
            seen_files.add(file_name)  # Mark file as processed

    response_txt = str(response)
    if file_links:
        response_txt += "<br>Reference files:<br>" + "<br>".join(file_links)
    if not highest_aggregated_score_file:  # If no file with a high score was found
        response_txt = llm.complete(query).text
    yield response_txt

def stream_chatbot(query, chat_history, session_id):
    if data_source == "nodataset":
        for response in call_llm_streamed(query):
            yield response
        return

    if is_chat_engine:
        response = engine.stream_chat(query)
    else:
        response = engine.query(query)

    partial_response = ""
    if len(response.source_nodes) == 0:
        response = llm.stream_complete(query)
        for token in response:
            partial_response += token.delta
            yield partial_response
    else:
        # Aggregate scores by file
        file_scores = defaultdict(float)
        for node in response.source_nodes:
            if 'filename' in node.metadata:
                file_name = node.metadata['filename']
                file_scores[file_name] += node.score

        # Find the file with the highest aggregated score
        highest_score_file = max(file_scores, key=file_scores.get, default=None)

        file_links = []
        seen_files = set()
        for token in response.response_gen:
            partial_response += token
            yield partial_response
            time.sleep(0.05)

        time.sleep(0.2)

        if highest_score_file:
            abs_path = Path(os.path.join(os.getcwd(), highest_score_file.replace('\\', '/')))
            file_name = os.path.basename(abs_path)
            file_name_without_ext = abs_path.stem
            if file_name not in seen_files:  # Check if file_name is already seen
                if data_source == 'youtube':
                    # Read and parse the XML file
                    try:
                        tree = ET.parse(abs_path)
                        root = tree.getroot()
                        title = root.find('title').text
                        channel = root.find('channel').text
                        upload_date = root.find('upload_date').text
                        youtube_info = f"<b>Title:</b> {title} <br> <b>Channel:</b> {channel} <br> <b>Upload Date:</b> {upload_date}"
                    except ET.ParseError:
                        youtube_info = "Failed to parse XML file"

                    file_link = f"<a href='https://www.youtube.com/watch?v={file_name_without_ext}'><img src='https://img.youtube.com/vi/{file_name_without_ext}/default.jpg'/></a>{youtube_info}<br>"
                elif data_source == 'directory':
                    file_link = f'<a href="file:////{abs_path}">{file_name}</a>'
                else:
                    exit("Wrong data_source type")
                file_links.append(file_link)
                seen_files.add(file_name)  # Add file_name to the set

        if file_links:
            partial_response += "<br>Reference files:<br>" + "<br>".join(file_links)
        yield partial_response

    # call garbage collector after inference
    torch.cuda.empty_cache()
    gc.collect()


interface = MainInterface(chatbot=stream_chatbot if streaming else chatbot, streaming=streaming)


def on_shutdown_handler(session_id):
    global llm, service_context, embed_model, faiss_storage, engine
    import gc
    if llm is not None:
        llm.unload_model()
        del llm
    # Force a garbage collection cycle
    gc.collect()


interface.on_shutdown(on_shutdown_handler)


def reset_chat_handler(session_id):
    global faiss_storage
    global engine
    print('reset chat called', session_id)
    if is_chat_engine == True:
        faiss_storage.reset_engine(engine)


interface.on_reset_chat(reset_chat_handler)


def on_dataset_path_updated_handler(source, new_directory, video_count, session_id):
    print('data set path updated to ', source, new_directory, video_count, session_id)
    global engine
    global data_dir
    if source == 'directory':
        if data_dir != new_directory:
            data_dir = new_directory
            generate_inferance_engine(data_dir)
    elif source == 'youtube':
        youtube_path = new_directory
        data_dir = os.path.join(os.getcwd(), 'youtube_dataset')
        status = fetch_transcripts(youtube_path, video_count)
        handle_regenerate_index(source, data_dir, session_id)
        generate_inferance_engine(data_dir)
        print("All videos transcripts are downloaded and processed") if status == True else print(
            "Not all videos transcripts are downloaded and processed")


interface.on_dataset_path_updated(on_dataset_path_updated_handler)


def on_model_change_handler(model, metadata, session_id):
    model_path = os.path.join(os.getcwd(), metadata.get('model_path', None))
    engine_name = metadata.get('engine', None)
    tokenizer_path = os.path.join(os.getcwd(), metadata.get('tokenizer_path', None))

    if not model_path or not engine_name:
        print("Model path or engine not provided in metadata")
        return

    global llm, embedded_model, engine, data_dir, service_context

    if llm is not None:
        llm.unload_model()
        del llm

    llm = TrtLlmAPI(
        model_path=model_path,
        engine_name=engine_name,
        tokenizer_dir=tokenizer_path,
        temperature=metadata.get('temperature', 0.1),
        max_new_tokens=metadata.get('max_new_tokens', 512),
        context_window=metadata.get('max_input_token', 512),
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False
    )
    service_context = ServiceContext.from_service_context(service_context=service_context, llm=llm)
    set_global_service_context(service_context)
    generate_inferance_engine(data_dir)


interface.on_model_change(on_model_change_handler)


def on_dataset_source_change_handler(source, path, session_id):

    global data_source, data_dir, engine
    data_source = source

    if data_source == "nodataset":
        print(' No dataset source selected', session_id)
        return
    
    print('dataset source updated ', source, path, session_id)
    
    if data_source == "youtube":
        data_dir = os.path.join(os.getcwd(), 'youtube_dataset')
    elif data_source == "directory":
        data_dir = path
    else:
        print("Wrong data type selected")
    generate_inferance_engine(data_dir)


interface.on_dataset_source_updated(on_dataset_source_change_handler)


def handle_regenerate_index(source, path, session_id):
    if data_source == "youtube":
        path = os.path.join(os.getcwd(), 'youtube_dataset')
    generate_inferance_engine(path, force_rewrite=True)
    print("on regenerate index", source, path, session_id)


interface.on_regenerate_index(handle_regenerate_index)
# render the interface
interface.render()
