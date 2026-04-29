














CHAPTER 1
 INTRODUCTION
 

1.1 Overview

Speech-based interaction has become a key component in modern AI systems. However, most systems rely on multi-stage processing pipelines, which introduce delays and reduce natural interaction.

This project aims to develop a real-time speech-to-speech system that eliminates intermediate bottlenecks and enables seamless communication.

1.2 Problem Statement

Existing conversational voice systems predominantly rely on a sequential processing pipeline, where speech input is first converted into text using Speech‑to‑Text (STT), then processed by a natural language or reasoning model, and finally transformed back into audio output via Text‑to‑Speech (TTS). While this approach is functionally effective, it introduces significant end‑to‑end latency, as each stage must complete before the next can begin. This batch‑oriented design disrupts conversational flow and results in delayed, unnatural interactions. Furthermore, such systems lack true real‑time audio streaming capabilities, preventing them from processing speech continuously or responding mid‑utterance. As a result, they struggle to support advanced conversational features such as interruption handling, overlapping dialogue, and efficient dynamic tool execution during live interactions. These limitations reduce the usability of traditional pipelines in scenarios that demand low‑latency, natural, and responsive voice‑based communication.

1.3 Objectives

The primary objective of this project is to design and implement a real‑time speech‑to‑speech conversational system that overcomes the latency and interaction constraints of traditional sequential pipelines. The system aims to enable continuous, low‑latency audio streaming by leveraging PCM‑based audio transmission and integrating the Azure Realtime Speech‑to‑Speech (S2S) model for live conversational intelligence. Another key objective is to support dynamic tool invocation during active conversations, allowing the model to call external tools and incorporate their results seamlessly into ongoing responses. The project also focuses on reducing response latency to improve conversational naturalness and ensuring robust support for multi‑turn interactions, enabling the system to maintain context and continuity across conversational exchanges. Together, these objectives seek to deliver a more responsive, human‑like voice interaction experience suitable for real‑world applications.

1.4 Scope

The scope of this project encompasses the development of a foundational real‑time conversational voice platform applicable across multiple domains. It is designed to support voice assistant systems that require immediate and natural responses, as well as AI‑powered customer support solutions where low latency and conversational continuity are critical for user satisfaction. The system is also relevant to healthcare conversational applications, such as virtual patient assistants or triage systems, where responsive voice interaction can enhance accessibility and usability. Additionally, the architecture is suitable for digital avatars and interactive agents, enabling lifelike spoken interaction in virtual environments. While the project focuses on the core speech‑to‑speech interaction pipeline and supporting mechanisms, it provides a scalable and extensible foundation for future enhancements and domain‑specific customization.

 

CHAPTER 2

BACKGROUND & LITERATURE REVIEW

2.1 Traditional Speech Systems

Traditional speech‑based conversational systems typically follow a sequential processing pipeline consisting of Speech Input, Speech‑to‑Text (STT), Text Processing, Text‑to‑Speech (TTS), and Audio Output. In this architecture, spoken input is first fully captured and converted into text before any language understanding or reasoning takes place. The generated textual response is then converted back into spoken audio for playback. While this approach simplifies system design and modularization, it introduces high end‑to‑end latency due to its strictly sequential execution. Each stage must complete before the next begins, resulting in noticeable delays between user input and system response. Additionally, converting audio into text early in the pipeline often leads to a loss of speech‑specific nuances, such as tone, emphasis, pauses, and emotional cues, which can negatively impact conversational quality and naturalness.

2.2 Real-Time Speech Systems

Modern real‑time speech systems aim to overcome the limitations of traditional pipelines by enabling continuous audio processing and streaming interaction. Instead of waiting for complete utterances, these systems process audio directly as it arrives, allowing input and output to be handled simultaneously. This streaming‑first approach significantly reduces perceived latency by enabling the system to begin generating responses while the user is still speaking. Real‑time speech systems support more natural conversational dynamics, including faster turn‑taking and smoother dialogue flow. By minimizing buffering and eliminating rigid sequential boundaries, these systems better approximate human conversational behavior and are particularly suited for applications requiring immediacy and responsiveness.

 

Modern systems:

Process audio directly
Stream input/output simultaneously
Reduce latency
2.3 Related Technologies

The development of conversational speech systems relies on advancements across several key technological areas. Speech recognition has been significantly improved by deep learning–based models such as Whisper, which enable robust transcription across diverse accents and acoustic conditions. Language modeling is primarily driven by transformer‑based architectures, which excel at understanding context, generating coherent responses, and reasoning across multi‑turn conversations. For speech synthesis, neural Text‑to‑Speech models have replaced traditional rule‑based approaches, enabling more natural, expressive, and human‑like audio generation. Together, these technologies form the foundation upon which both traditional and modern conversational speech systems are built.

Speech Recognition

Whisper models
Language Models

Transformer-based models
Speech Synthesis

Neural TTS
 

2.4 Gap Analysis

Despite advances in speech recognition, language modeling, and speech synthesis, many existing conversational systems still fail to deliver truly real‑time interaction. Most systems remain turn‑based rather than fully duplex, meaning they cannot listen and respond simultaneously. This limitation prevents interruption handling and dynamic conversational flow. Furthermore, integration of external tools and services during live conversations is often limited or inefficient, as tool invocation is typically performed after a full input and reasoning cycle completes. These gaps highlight the need for architectures that support continuous streaming, full‑duplex interaction, and seamless tool execution in real time, motivating the design choices explored in this project.

 

Existing systems:

Not real-time
Not full-duplex
Limited tool integration
CHAPTER 3

 SYSTEM ARCHITECTURE

3.1 Overall Architecture

The system architecture is designed to support real‑time, bidirectional speech‑to‑speech interaction by integrating a browser‑based frontend, a streaming backend, an AI realtime processing layer, and an external tool execution layer. The architecture follows a modular design that separates audio capture, communication, reasoning, and tool execution, enabling scalability and maintainability.

The frontend serves as the user interaction layer and communicates with the backend through a persistent WebSocket connection. The backend acts as the central coordination layer, managing audio streaming, session handling, and message routing. It forwards streaming audio input to the AI model through a realtime API and simultaneously handles streaming audio output returned by the model. The AI model processes the incoming audio stream, generates responses in real time, and can dynamically invoke external tools when additional information or computation is required. These tool invocations are handled by a dedicated tool layer, whose results are fed back into the AI model to produce final spoken responses.

This architecture enables low‑latency, continuous interaction by avoiding batch processing and allowing input, processing, and output to occur concurrently.

 

System Architecture Diagram
+-------------+      +-------------+      +----------------+
|  Frontend   | ---> |   Backend   | ---> |  AI Model      |
| (Audio I/O) | <--- | (WebSocket) | <--- | (Realtime API) |
+-------------+      +-------------+      +----------------+
                           |
                           v
                     +-----------+
                     | Tool Layer|
                     +-----------+

3.2 Components

1. Frontend

The frontend component is responsible for all user‑facing audio interactions. It captures microphone input from the user using browser‑based audio APIs and processes the raw audio into a format suitable for real‑time streaming. Specifically, the recorded audio is converted from floating‑point samples into PCM‑16 format, which is widely supported by realtime speech models and optimized for low‑latency transmission. The frontend sends audio chunks continuously to the backend over a WebSocket connection. Additionally, it receives streaming audio responses from the backend, decodes them, and plays them immediately to the user. This enables smooth, uninterrupted playback and contributes to a natural conversational experience.

Captures audio
Converts Float32 → PCM16
Plays output audio
 

2. Backend

The backend acts as the orchestration and communication layer of the system. It maintains persistent WebSocket connections with frontend clients and manages real‑time audio streaming sessions. Incoming audio chunks from the frontend are buffered briefly to maintain packet order and timing consistency before being forwarded to the AI model through the realtime API. The backend also receives streaming audio and text responses from the model and routes them back to the frontend with minimal delay. In addition to audio handling, the backend manages session state, coordinates tool execution requests, and ensures reliable communication between system components. This intermediary role allows the backend to isolate frontend and model logic while enforcing control over data flow and system stability.

Handles WebSocket communication
Buffers audio chunks
Sends data to model
3. AI Model

The AI model is responsible for real‑time speech understanding, reasoning, and response generation. It processes streaming audio input directly and maintains conversational context throughout the session. Unlike traditional systems that require complete utterances, the model operates incrementally, enabling it to begin response generation as soon as sufficient information is available. During a conversation, the model can dynamically request the execution of external tools, such as data retrieval or calculation services, when additional information is needed. Once tool results are returned, the model incorporates them into the ongoing response and continues generating audio output. This capability enables intelligent, context‑aware interactions within a single continuous conversation flow.

Processes streaming audio
Generates responses
Calls tools dynamically
 

 

4. Tool Layer

The tool layer provides access to external services and functions that extend the model’s capabilities beyond natural language reasoning. Examples include weather information retrieval, mathematical calculations, and other application‑specific services. When the AI model determines that a tool is required, the backend invokes the appropriate tool through this layer and returns the result to the model in a structured format. By separating tool execution from the core model logic, the system maintains a clean architecture while enabling flexible integration of new tools. This design allows the conversational agent to provide accurate, up‑to‑date, and actionable responses during live interactions without disrupting the real‑time conversational flow.

Weather API
Calculator
External services
 

CHAPTER 4

METHODOLOGY

4.1 Audio Processing

Audio processing is a critical component of the proposed real‑time speech‑to‑speech system, as it directly affects latency and interaction quality. Audio input captured from the frontend microphone is initially represented in Float32 format, which is commonly used by browser audio APIs due to its high precision. However, this format is not optimal for real‑time transmission or model compatibility. Therefore, the audio data is converted into PCM‑16 format, a lightweight and widely supported representation that balances quality and efficiency. The system supports sampling rates of 16 kHz and 24 kHz, enabling flexibility depending on model requirements and performance considerations. To ensure safe and consistent transmission over WebSocket connections, the processed audio chunks are encoded using Base64 encoding, allowing binary data to be embedded within standard text‑based messages.

Input: Float32
Converted to PCM16
Sample Rate: 16kHz / 24kHz
Encoded using Base64
4.2 Streaming Pipeline

The streaming pipeline is designed to enable continuous, low‑latency audio transmission from the user to the AI model. The pipeline begins with real‑time audio capture at the frontend, where the microphone input is continuously recorded. The captured audio is divided into small, time‑bounded chunks to ensure smooth streaming and minimal buffering delay. Each chunk is then Base64‑encoded and transmitted to the backend through a persistent WebSocket connection. The backend forwards these audio chunks to the AI model in near real time. This streaming‑oriented pipeline avoids batch processing and ensures that audio input, processing, and response generation can occur concurrently, forming the foundation for natural conversational interaction.

Audio Capture → Chunking → Base64 Encoding → WebSocket → Model

4.3 Event Flow

Communication between the backend and the AI model follows a structured event‑driven protocol. As audio chunks arrive, they are sent to the model using input_audio_buffer.append events, allowing the model to process incoming speech incrementally. Once the user finishes speaking, an input_audio_buffer.commit event is issued to signal the completion of the current utterance. This commit triggers the model to finalize its interpretation of the input and begin response generation, which is initiated through a response.create event. The model then streams its output back to the backend as a sequence of incremental updates, including audio.delta events for synthesized speech output and text.delta events for partial or complete textual responses. This event flow enables real‑time feedback while maintaining clear conversational boundaries.

input_audio_buffer.append
input_audio_buffer.commit
response.create
Receive:
audio.delta
text.delta
4.4 Tool Calling Workflow

The system supports dynamic tool invocation as part of the conversational flow, enabling the AI model to access external information or perform computations during live interactions. When a user provides input, the model analyzes the request and determines whether a tool is required to generate an accurate response. If a tool is needed, the model emits a structured tool‑call request identifying the appropriate function and associated parameters. The backend then executes the requested tool through the tool layer and returns the result to the model. The model incorporates this result into its reasoning process and continues generating the final response, which is streamed back to the user in audio form. This workflow allows tools to be integrated seamlessly without interrupting the conversational experience.

User Input → Model → Tool Detection → Tool Execution → Response Generation

4.5 Latency Optimization Techniques

Several targeted optimization techniques were employed to reduce latency and improve real‑time responsiveness. The use of compressed codecs such as WebM was avoided due to their encoding and decoding overhead. Instead, raw PCM‑16 audio was adopted to minimize processing delay. Audio data is streamed in small chunks rather than waiting for complete utterances, enabling early processing and faster response initiation. Additionally, the system maintains persistent sessions with the AI model, eliminating the overhead associated with repeated connection setup and teardown. Together, these optimizations significantly reduce end‑to‑end latency and contribute to a smoother, more natural conversational experience.

Removed WebM codec
Used PCM16 format
Streaming chunks
Persistent session
 

 

 

CHAPTER 5

IMPLEMENTATION

5.1 Backend Implementation

The backend of the system is implemented using a Flask‑based server, which serves as the central coordination layer for real‑time communication, session management, and model interaction. Flask was chosen for its lightweight nature and flexibility in handling asynchronous workflows. The backend establishes and maintains persistent WebSocket connections with frontend clients to support continuous bi‑directional communication required for real‑time audio streaming. Through these WebSocket connections, the backend receives streaming audio input, forwards it to the AI model, and simultaneously streams model responses back to the frontend. Integration with the Azure Realtime SDK enables the backend to manage realtime model sessions, transmit audio events, receive streaming responses, and coordinate tool invocation. This backend implementation acts as the backbone of the system, ensuring low‑latency data flow, session stability, and reliable orchestration across all components.

Flask server
WebSocket handling
Azure SDK integration
 

5.2 Frontend Implementation

The frontend is implemented using HTML and JavaScript, leveraging standard browser technologies to ensure broad compatibility and ease of deployment. The Web Audio API is used to capture microphone input from the user in real time and process the raw audio data for streaming. Audio samples are prepared for transmission by converting them into a model‑compatible format before being streamed to the backend through a WebSocket connection. The frontend is also responsible for real‑time playback of the synthesized audio responses received from the backend. Incoming audio is decoded and played immediately to minimize perceived latency and maintain conversational fluidity. This implementation enables users to interact with the system using natural voice input while receiving uninterrupted spoken responses.

HTML + JavaScript
Web Audio API
Real-time playback
 

5.3 Tool Integration

The system includes a dedicated mechanism for integrating external tools that enhance the conversational capabilities of the AI model. Example tools implemented in this project include a weather information service and a calculator, which allow the system to retrieve real‑world data and perform computations during live conversations. When the AI model determines that external information or processing is required, it issues a tool request that is handled by the backend. The backend executes the requested tool and returns the result to the model in a structured format. This approach enables dynamic tool usage without interrupting the real‑time conversational flow and allows additional tools and services to be integrated in the future with minimal changes to the overall architecture.

Example tools:

Weather API
Calculator
5.4 Key Features

The implemented system provides several key features that distinguish it from traditional conversational voice systems. Most notably, it supports real‑time audio streaming, enabling continuous speech input and output with
