# No JS LLM App

A streamlined, FastAPI-based web application for interacting with Large Language Models (LLMs), built without JavaScript. This application is designed for developers and organizations seeking a minimalistic, efficient, and easily maintainable solution.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Ollama Installation and Setup](#ollama-installation-and-setup)
  * [Redis Installation and Setup](#redis-installation-and-setup)
  * [Configuration](#configuration)
  * [Running the Application](#running-the-application)
* [Testing](#testing)
* [Project Structure](#project-structure)
* [Customization](#customization)
* [Deployment](#deployment)
* [Continuous Integration](#continuous-integration)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Overview

The No JS LLM App enables interaction with Large Language Models through a Python FastAPI backend, entirely eliminating the use of JavaScript. It emphasizes simplicity, reliability, and straightforward integration, making it ideal for educational purposes, rapid prototyping, and production deployment.

---

## Features

* **JavaScript-Free Frontend:** Built purely with HTML and CSS for fast, lightweight performance.
* **Session Management:** Securely maintains conversation contexts across user interactions.
* **Rate Limiting:** Built-in protection against excessive or abusive use.
* **Docker Support:** Containerized setup for consistent deployment.
* **Automated Testing:** Includes pytest-based tests ensuring code quality and stability.
* **CI Integration:** GitHub Actions workflow automates testing and validation processes.

---

## Architecture Overview

Here’s how the components of the No JS LLM App interact:

```
+-----------------------+                +---------------------------+               +------------------+
|  User's Browser (UI)  |   HTTP Form    |   FastAPI App (Python)    |   SQL Queries |     SQLite DB    |
|  HTML & CSS Only UI   +--------------->|  Renders HTML, Handles API+---------------> (Session Store)   |
|  No JS, Lightweight   |                |  logic via /chat, /stream |               +------------------+
+-----------------------+                +---------------------------+                        ^
        |                                          |                                           |
        |                                          | Redis Cache (rate limit)                 |
        |                                          v                                           |
        |                                 +------------------+         REST Call              |
        |                                 |      Redis       |<-------------------------------+
        |                                 | Rate Limiter via |
        |                                 | fastapi-limiter  |
        |                                 +------------------+
        |                                          |
        |                                          |
        |                              REST API    |  Ollama API Request
        |                                          v
        |                                +--------------------+
        |                                |   Ollama LLM Host   |
        |                                |  (Qwen3 or Gemma3)  |
        |                                +--------------------+
        |                                          |
        |                          Streamed/Text Response
        +<-----------------------------------------+
```

**Technology Layers (Left to Right):**

* **Frontend (HTML/CSS)**: Stateless client form that sends user inputs to backend via HTTP.
* **FastAPI App**: Backend app that serves the frontend, processes LLM prompts, and enforces rate limits.
* **Redis**: In-memory key-value store for managing per-user request throttling.
* **SQLite**: Local persistent database for saving session messages.
* **Ollama**: LLM runtime that executes model inference (qwen3 or gemma3).

Use `/chat` for full response polling, or `/chat-stream` for real-time response streaming.

```
+------------------+          +------------------+          +------------------+
|  User's Browser  |  <--->   |   FastAPI App    |  <---->  |     SQLite DB    |
| (HTML + CSS UI) |  Form     |  (/chat or /chat-stream)    |  (Session Store) |
+------------------+          +------------------+          +------------------+
        |                               |                             
        |                               | REST API                   
        |                               v                             
        |                      +------------------+                  
        |                      |     Ollama LLM    |                 
        |                      | (qwen3 or gemma3) |                 
        |                      +------------------+                 
        |                               ^                             
        |     Rate Limit via Redis       |                             
        |     +------------------+       |                             
        +-->  |      Redis        | <----+                             
             +------------------+                                       
```

* User submits prompts via an HTML form (no JS required).
* FastAPI handles the request and routes it to `/chat` or `/chat-stream`.
* Based on the model selected (qwen3:1.7b or gemma3:4b), it sends the prompt to Ollama.
* The response is returned to the user, while rate limits and session data are enforced via Redis and SQLite.

---

## Tech Stack

* **FastAPI**: Web Framework – FastAPI is chosen for its asynchronous capabilities, automatic validation, and built-in support for interactive docs. It is more modern and faster than Flask for handling concurrent requests.

  * *Alternative:* Flask – more beginner-friendly, but slower and lacks async support.

* **Python 3.12**: Backend Development – Python is selected for its simplicity, huge library ecosystem, and native support for data handling and AI integration.

  * *Alternative:* Node.js or Go – both efficient but lack the deep AI tooling Python offers.

* **SQLite**: Lightweight Database – Used to store session history with minimal configuration.

  * *Alternative:* PostgreSQL – more powerful but overkill for a lightweight single-user or demo app.

* **Docker & Docker Compose**: Containerization – Enables reproducible environments, great for cross-platform deployment.

  * *Alternative:* Virtualenv or system Python – less portable.

* **HTML & CSS**: Frontend Design – Plain HTML + CSS keep the interface lightweight and accessible.

  * *Alternative:* React or Vue.js – powerful but requires JS runtime.

* **Pytest**: Testing Framework – For writing and running unit/integration tests.

  * *Alternative:* unittest – built-in but more verbose.

* **Ollama**: Local LLM Management – Runs and manages LLMs like Qwen3 or Gemma locally. Fully compatible with Mac, Windows, Linux

  * *Alternative:* LLM Studio (not fully compatible with Mac. Requires additional setup)

* **Redis**: Rate Limiting Cache – Handles real-time request throttling and session memory.

  * *Alternative:* In-memory Python dict or PostgreSQL – either too volatile or too heavy.

---

## LLM Model Selection

We use two models in this app, carefully chosen for their balance of performance and resource-efficiency:

* **Qwen3:1.7B** – A compact, high-performing model optimized for reasoning and summarization tasks.

  * *Why this model?* It delivers surprisingly good performance given its size and runs well on laptops with less than 8GB RAM.
  * *Benchmark:* In recent benchmarks (2024), Qwen3:1.7B outperformed many other sub-2B models on common-sense reasoning and instruction following.
  * *Alternative:* Mistral 7B – more accurate but significantly heavier, requires higher RAM and GPU specs.

* **Gemma3:4B** – A lightweight multimodal model capable of processing both text and images.

  * *Why this model?* Perfect for simple automation tasks like turning scanned receipts or handwritten notes into structured data.
  * *Alternative:* Llama3.2 Vision 11B – size too large

---

## Getting Started

### Prerequisites

* Python 3.12 or higher
* Docker and Docker Compose
* Git (recommended)

### Installation

Clone this repository and navigate into the directory:

```bash
git clone <your_repository_url>
cd ai-project-1
```

Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Ollama Installation and Setup

Install Ollama either locally or via Docker to manage your LLMs:

* **Local Installation:** Follow the [Ollama documentation](https://github.com/ollama/ollama).

* **Docker Installation:**

```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
```

#### Add Models

Add recommended LLM models:

```bash
ollama pull qwen3:1.7b
ollama pull gemma3:4b
```

Ensure Ollama is accessible at `http://localhost:11434`.

### Redis Installation and Setup

Install Redis locally or via Docker to enable rate limiting functionality:

* **Docker Installation:**

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

Ensure Redis is running and accessible at `localhost:6379`.

### Configuration

Copy the `.env.example` file and customize settings:

```bash
cp .env.example .env
```

Edit `.env` with your database paths, API keys, Redis configuration, and Ollama configuration.

### Running the Application

#### Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Visit [http://localhost:8000](http://localhost:8000).

#### Docker

```bash
docker compose up --build
```

Visit [http://localhost:8000](http://localhost:8000).

---

## Testing

Run automated tests:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Project Structure

```
ai-project-1/
├── app/
│   ├── main.py                 # FastAPI app logic
│   ├── settings.py             # Configuration settings
│   ├── templates/              # HTML UI templates
│   └── static/                 # Static CSS files
├── tests/                      # Automated tests
│   ├── test_main.py            # Tests API routes including /chat and /chat-stream
│   ├── test_sessions.py        # Verifies session creation, retrieval, and isolation logic
│   ├── test_rate_limit.py      # Checks that the rate limiting (via Redis) is correctly enforced
│   └── conftest.py             # Test configuration and reusable fixtures setup
├── Dockerfile                  # Docker build file
├── docker-compose.yml          # Docker Compose config
├── requirements.txt            # Main dependencies
├── requirements-dev.txt        # Dev dependencies
├── .env.example                # Example configuration
└── .github/workflows/ci.yml    # GitHub CI workflow
```

---

## Customization

Easily extend functionality by:

* Modifying frontend templates and styles (`app/templates`, `app/static`).
* Updating backend logic (`app/main.py`).
* Adjusting or extending the database schema.
* **Enable real-time LLM output streaming** by editing the HTML form:

  * Open `app/templates/index.html`
  * Locate the form block:

    ```html
    <form action="/chat" method="post">
    <!-- <form action="/chat-stream" method="post"> -->
    ```
  * To switch to streaming mode, comment out the `/chat` form and uncomment the `/chat-stream` form:

    ```html
    <!-- <form action="/chat" method="post"> -->
    <form action="/chat-stream" method="post">
    ```
  * This activates the streaming endpoint so users can see LLM responses as they are being generated.

---

## Deployment

Deploy using Docker Compose:

```bash
docker compose -f docker-compose.yml up -d --build
```

For production environments, consider deploying behind a reverse proxy like Nginx or Apache.

---

## Continuous Integration

GitHub Actions workflow for automatic testing is found in `.github/workflows/ci.yml`.

---

## Contributing

Your contributions are welcome:

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to your branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Thanks to the open-source community and all contributors for continuous improvements.
