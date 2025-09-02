pipeline {
  agent any

  environment {
    IMAGE_NAME = 'ars-gui-app'
    IMAGE_TAG  = 'latest'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Validate Python') {
      steps {
        sh 'python3 --version || python --version || true'
        sh 'pip3 --version || pip --version || true'
      }
    }

    stage('Build Docker Image') {
      steps {
        script {
          sh 'docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .'
        }
      }
    }

    // Optional: Build a standalone binary with PyInstaller in a Python container
    stage('PyInstaller Build (optional)') {
      when {
        expression { return fileExists('5.Final_GUI.spec') || fileExists('5.Final_GUI.py') }
      }
      steps {
        sh '''
          docker run --rm -v "$PWD":/work -w /work python:3.10-slim /bin/bash -lc "
            apt-get update && apt-get install -y --no-install-recommends python3-tk tk gcc g++ make &&             pip install --no-cache-dir -r requirements.txt pyinstaller &&             if [ -f 5.Final_GUI.spec ]; then               pyinstaller 5.Final_GUI.spec;             else               pyinstaller --onefile --windowed                 --add-data 'ARS.png:.'                 --add-data 'classifier_model.tflite:.'                 --add-data 'efficientnetb3_notop.h5:.'                 --add-data 'regression_model.tflite:.'                 --add-data 'standard_scaler.pkl:.'                 5.Final_GUI.py;             fi
          "
        '''
      }
    }

    // Optional: Push to a registry (uncomment and configure credentials & registry)
    // stage('Docker Login & Push') {
    //   when { expression { return env.DOCKER_REGISTRY && env.DOCKER_CREDENTIALS_ID } }
    //   steps {
    //     withCredentials([usernamePassword(credentialsId: "${DOCKER_CREDENTIALS_ID}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
    //       sh "echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin ${DOCKER_REGISTRY}"
    //       sh "docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    //       sh "docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    //     }
    //   }
    // }

    stage('Archive Artifacts') {
      when { expression { return fileExists('dist') } }
      steps {
        archiveArtifacts artifacts: 'dist/**', fingerprint: true
      }
    }
  }

  options {
    timestamps()
    ansiColor('xterm')
  }
}
