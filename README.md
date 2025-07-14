# No JS LLM App

A streamlined, Flask-based Large Language Model (LLM) web application designed without JavaScript, ideal for developers and organizations seeking a minimalistic, highly efficient, and easily maintainable solution for interacting with language models.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
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

The No JS LLM App is a Python Flask-based web solution enabling interaction with Large Language Models without any reliance on JavaScript. This project focuses on simplicity, reliability, and ease of integration, making it an excellent choice for educational purposes, prototyping, and production deployments.

---

## Features

* **No JavaScript:** Pure HTML/CSS frontend implementation for optimal simplicity and performance.
* **Session Management:** Maintain conversational context and manage user sessions securely.
* **Rate Limiting:** Protect your application from excessive use with built-in rate limiting.
* **Dockerized:** Easy setup and deployment using Docker.
* **Testing Framework:** Includes automated tests using pytest to ensure reliability.
* **Continuous Integration:** Integrated GitHub Actions workflow for automated testing and quality assurance.

---

## Tech Stack

| Technology       | Purpose              |
| ---------------- | -------------------- |
| Flask            | Web Framework        |
| Python 3.12      | Backend Development  |
| SQLite           | Lightweight Database |
| Docker & Compose | Containerization     |
| HTML & CSS       | Frontend Design      |
| Pytest           | Testing Framework    |

---

## Getting Started

### Prerequisites

* Python 3.12 or higher
* Docker and Docker Compose
* Git (recommended)

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone <your_repository_url>
cd ai-project-1
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and configure necessary settings:

```bash
cp .env.example .env
```

Adjust `.env` variables such as database paths, API keys, and other configurations as required.

### Running the Application

#### Locally

Run the Flask application locally:

```bash
flask --app app/main.py run --debug
```

Open your browser and access the application at [http://localhost:5000](http://localhost:5000).

#### Docker

Build and run using Docker Compose:

```bash
docker compose up --build
```

Visit [http://localhost:5000](http://localhost:5000) to interact with the application.

---

## Testing

Run automated tests to ensure application reliability:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Project Structure

```
ai-project-1/
├── app/
│   ├── main.py                 # Flask application logic
│   ├── settings.py             # Configuration settings
│   ├── templates/              # HTML files for UI
│   └── static/                 # Static assets (CSS)
├── tests/                      # Automated test cases
├── Dockerfile                  # Docker build instructions
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Development dependencies
├── .env.example                # Example environment configuration
└── .github/workflows/ci.yml    # CI workflows for GitHub Actions
```

---

## Customization

Customize and extend this application by:

* Modifying HTML/CSS files within `app/templates` and `app/static`.
* Enhancing backend logic in `app/main.py`.
* Extending the database schema for additional functionality.

---

## Deployment

Deploy easily with Docker Compose in a production environment:

```bash
docker compose -f docker-compose.yml up -d --build
```

For more robust deployments, consider reverse proxy solutions such as Nginx or Apache.

---

## Continuous Integration

Integrated GitHub Actions automate testing and validation, ensuring stable deployments. View or modify the CI workflow in `.github/workflows/ci.yml`.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments

Special thanks to the open-source community and contributors whose insights and suggestions continually help improve this project.
