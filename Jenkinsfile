// Declarative Jenkins Pipeline
pipeline {
    agent any

    // Environment variables can be defined here if needed
    environment {
        // Define the name for the Docker image
        DOCKER_IMAGE_NAME = 'hki-jena/ars-gui'
        // Define a tag for the image, using the build number for versioning
        DOCKER_IMAGE_TAG = "build-${BUILD_NUMBER}"
    }

    stages {
        // Stage 1: Checkout code from the GitLab repository
        stage('Checkout') {
            steps {
                script {
                    echo 'Checking out code from GitLab...'
                    // This command checks out the code from your repository.
                    // NOTE: You would need to configure Jenkins with credentials
                    // to access this private GitLab repository.
                    git url: 'https://asb-git.hki-jena.de/applied-systems-biology/ars-gui.git', branch: 'main'
                }
            }
        }

        // Stage 2: Build the Docker image
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
                    // Build the Docker image using the Dockerfile in the workspace
                    docker.build("${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}", '.')
                }
            }
        }

        // Stage 3: Push Docker Image (Optional)
        // This stage is for pushing the built image to a Docker registry.
        // It's commented out as it requires a configured registry.
        /*
        stage('Push Docker Image') {
            steps {
                script {
                    // Example for pushing to a Docker registry like Docker Hub or a private one.
                    // You would need to configure Jenkins with registry credentials.
                    docker.withRegistry('https://your-docker-registry.com', 'your-registry-credentials-id') {
                        echo "Pushing image ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}..."
                        docker.image("${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}").push()
                    }
                }
            }
        }
        */

        // Stage 4: Run Application (Placeholder)
        // Running a GUI application from Jenkins is not a standard use case,
        // as Jenkins is typically for automated, non-interactive tasks.
        // This stage is a placeholder to show where you might run automated tests.
        stage('Run/Test') {
            steps {
                script {
                    echo 'This stage is a placeholder.'
                    echo 'You could run automated GUI tests here if you have a framework like PyAutoGUI.'
                    // Example of how you might run the container (requires a display):
                    // sh "docker run -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
                }
            }
        }
    }

    // Post-build actions, such as cleaning up the workspace
    post {
        always {
            echo 'Pipeline finished. Cleaning up...'
            cleanWs() // Deletes the workspace after the build
        }
    }
}