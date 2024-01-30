import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import sounddevice as sd
from scipy.io.wavfile import write
import base64
import hashlib
import hmac
import os
import time
import requests
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components

# Load emotion model
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
holis = holistic.Holistic()

# Streamlit settings
st.set_page_config(
    page_title="Music Recommendation App",
    layout="wide",
    page_icon="🎵",
)

# Set background color and padding
st.markdown(
    """
    <style>
        body {
            background-color: #f0f5f5;
        }
        .stApp {
            padding: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with options
option = st.sidebar.selectbox(
    "Choose an option",
    ["Emotion-based Music Recommender", "Audio Identification", "Song Recommendation"],
)

# Emotion-based Music Recommender
if option == "Emotion-based Music Recommender":
    st.header("Emotion-Based Music Recommender")
    if "run" not in st.session_state:
        st.session_state["run"] = "true"

    try:
        emotion = np.load("emotion.npy")[0]
    except:
        emotion=""

    if not(emotion):
        st.session_state["run"] = "true"
    else:
        st.session_state["run"] = "false"


    class EmotionProcessor:
        def recv(self, frame):
            frm = frame.to_ndarray(format="bgr24")

            ##############################
            frm = cv2.flip(frm, 1)

            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            lst = []

            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                lst = np.array(lst).reshape(1,-1)

                pred = label[np.argmax(model.predict(lst))]

                print(pred)
                cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

                np.save("emotion.npy", np.array([pred]))

                
            return av.VideoFrame.from_ndarray(frm, format="bgr24")
        
    lang = st.text_input("Language")
    singer = st.text_input("singer")

    if lang and singer:
        # Configure RTC (Real-Time Communication) settings with video constraints
        rtc_configuration = RTCConfiguration({"video": {"width": 640, "height": 480}})

        webrtc_streamer(
            key="key",
            rtc_configuration=rtc_configuration,
            desired_playing_state=True,
            video_processor_factory=EmotionProcessor,
        )

        btn = st.button("Recommend me songs")

        if btn:
            emotion = np.load("emotion.npy")[0]
            if not emotion:
                st.warning("Please let me capture your emotion first")
            else:
                webbrowser.open(
                    f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}"
                )
                np.save("emotion.npy", np.array([""]))


# Audio Identification
elif option == "Audio Identification":
    st.header("Audio Identification")

    # Function to identify audio
    def identify_audio():
        try:
            # Audio recording and identification logic
            filename = "recorded_audio.wav"
            duration = 15
            samplerate = 44100

            st.write("Recording...")
            audio_data = sd.rec(
                int(samplerate * duration), samplerate=samplerate, channels=2, dtype="int16"
            )
            sd.wait()
            st.write("Recording complete.")

            st.write("Saving to", filename)
            write(filename, samplerate, audio_data)

            # Replace "###...###" with your ACRCloud API credentials
            access_key = "0334bedf1125b4de6a26d13dca71f5a0"
            access_secret = "5vZIoelzRRJk0fPAyPc2j7WcHn0E6gLLpHzfxKwb"
            requrl = "https://identify-eu-west-1.acrcloud.com/v1/identify"

            http_method = "POST"
            http_uri = "/v1/identify"
            data_type = "audio"
            signature_version = "1"
            timestamp = time.time()

            string_to_sign = (
                http_method
                + "\n"
                + http_uri
                + "\n"
                + access_key
                + "\n"
                + data_type
                + "\n"
                + signature_version
                + "\n"
                + str(timestamp)
            )

            sign = base64.b64encode(
                hmac.new(
                    access_secret.encode("ascii"),
                    string_to_sign.encode("ascii"),
                    digestmod=hashlib.sha1,
                ).digest()
            ).decode("ascii")

            # Check if the specified file exists
            if not os.path.isfile(filename):
                st.error(f"Error: File '{filename}' not found.")
                return

            sample_bytes = os.path.getsize(filename)

            files = {"sample": (os.path.basename(filename), open(filename, "rb"), "audio/mpeg")}
            data = {
                "access_key": access_key,
                "sample_bytes": sample_bytes,
                "timestamp": str(timestamp),
                "signature": sign,
                "data_type": data_type,
                "signature_version": signature_version,
            }

            r = requests.post(requrl, files=files, data=data)
            r.encoding = "utf-8"
            response_json = r.json()

            if "metadata" in response_json:
                metadata = response_json["metadata"]
                if "music" in metadata:
                    music_info_list = metadata["music"]
                    if music_info_list:
                        first_result = music_info_list[0]
                        st.success("Identification successful")

                        # Display song name and artist name
                        st.write(f"Song Name: {first_result['title']}")
                        st.write(
                            f"Artist: {', '.join(artist['name'] for artist in first_result['artists'])}"
                        )

                        # Create a YouTube link based on the identified song
                        youtube_search_query = (
                            f"{first_result['title']} {first_result['artists'][0]['name']} official music video"
                        )
                        st.write("Listen on YouTube:")
                        st.markdown(
                            f"[{first_result['title']} - {first_result['artists'][0]['name']}]"
                            "(https://www.youtube.com/results?search_query={youtube_search_query})",
                            unsafe_allow_html=True,
                        )
                        webbrowser.open(
                            f"https://www.youtube.com/results?search_query={youtube_search_query}"
                        )
                    else:
                        st.error({"success": False, "message": "No music information found in the response."})
                else:
                    st.error({"success": False, "message": "No music information found in the response."})
            else:
                st.error({"success": False, "message": "No metadata found in the response."})

        except Exception as e:
            st.error({"success": False, "error": f"Error during API request: {e}"})

    if st.button("Identify Audio"):
        identify_audio()

# Song Recommendation
elif option == "Song Recommendation":
    st.header("Song Recommendation Engine")

    # Initialize session state
    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = []

    @st.cache(allow_output_mutation=True)
    def load_data():
        df = pd.read_csv("data/filtered_track_df.csv")
        df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
        exploded_track_df = df.explode("genres")
        return exploded_track_df

    genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
    audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

    exploded_track_df = load_data()

    def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
        genre = genre.lower()
        genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
        genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

        neigh = NearestNeighbors()
        neigh.fit(genre_data[audio_feats].to_numpy())

        n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]

        uris = genre_data.iloc[n_neighbors]["uri"].tolist()
        audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
        return uris, audios

    title = "Song Recommendation Engine"
    st.title(title)

    st.write("Welcome! Customize your music preferences and discover new songs.")
    st.markdown("##")

    with st.container():
        col1, col2, col3, col4 = st.columns([2, 0.5, 0.5, 0.5])
        with col3:
            st.markdown("***Choose your genre:***")
            genre = st.radio(
                "",
                genre_names, index=genre_names.index("Pop"))
        with col1:
            st.markdown("***Choose features to customize:***")
            start_year, end_year = st.slider(
                'Select the year range',
                1990, 2019, (2015, 2019)
            )
            acousticness = st.slider(
                'Acousticness',
                0.0, 1.0, 0.5)
            danceability = st.slider(
                'Danceability',
                0.0, 1.0, 0.5)
            energy = st.slider(
                'Energy',
                0.0, 1.0, 0.5)
            instrumentalness = st.slider(
                'Instrumentalness',
                0.0, 1.0, 0.0)
            valence = st.slider(
                'Valence',
                0.0, 1.0, 0.45)
            tempo = st.slider(
                'Tempo',
                0.0, 244.0, 118.0)

    tracks_per_page = 6
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)

    tracks = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
        tracks.append(track)

    current_inputs = [genre, start_year, end_year] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        if st.button("Recommend More Songs"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                if i % 2 == 0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See more details"):
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)

                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See more details"):
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)

        else:
            st.write("No songs left to recommend")
