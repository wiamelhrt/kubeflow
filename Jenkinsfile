pipeline {
    agent any
    stages {
        stage('Clone Repository') {
            steps {
                checkout scm
            }
        }
        
        stage('kubeflow pipeline') {
            steps {
                bat 'python kubeflow.py'
               
            }
        }
    }
}
