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
                bat '''kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80 
'''
            }
        }
    }
}
