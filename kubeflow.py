import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore") 
import kfp
import kfp.components as comp
import requests
import kfp.dsl as dsl

def prepare_data():
    import pandas as pd
    
    print("---- Inside prepare_data component ----")
    df = pd.read_csv("https://raw.githubusercontent.com/pik1989/MarketSegmentation/main/Customer%20Data.csv")
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())
    df = df.drop(columns=["CUST_ID"],axis=1)
    df.to_csv(f'data/final_df.csv', index=False)
    print("\n ---- data csv is saved to PV location /data/final_df.csv ----")

def model_building():
    import pandas as pd
    import sklearn
    from sklearn.preprocessing import StandardScaler
    scalar=StandardScaler()
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
    print("---- Inside model building ----")
    df = pd.read_csv(f'data/final_df.csv')
    
    scaled_df = scalar.fit_transform(df)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2"])
    kmeans_model=KMeans(4)
    kmeans_model.fit_predict(scaled_df)
    pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)
    # Creating a target column "Cluster" for storing the cluster segment
    cluster_df = pd.concat([df,pd.DataFrame({'Cluster':kmeans_model.labels_})],axis=1)
    cluster_1_df = cluster_df[cluster_df["Cluster"]==0]
    cluster_2_df = cluster_df[cluster_df["Cluster"]==1]
    cluster_3_df = cluster_df[cluster_df["Cluster"]==2]
    cluster_4_df = cluster_df[cluster_df["Cluster"] == 3]
    cluster_df.to_csv(f'data/clustered_df.csv', index=False)
    
def train_test_split():
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.model_selection import train_test_split
    print("---- Inside train_test_split component ----")
    cluster_df = pd.read_csv(f'data/clustered_df.csv')
    #Split Dataset
    X = cluster_df.drop(['Cluster'],axis=1)
    y= cluster_df[['Cluster']]
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3)
    
    np.save(f'data/X_train.npy', X_train)
    np.save(f'data/X_test.npy', X_test)
    np.save(f'data/y_train.npy', y_train)
    np.save(f'data/y_test.npy', y_test)
    
    print("\n---- X_train ----")
    print("\n")
    print(X_train)
    
    print("\n---- X_test ----")
    print("\n")
    print(X_test)
    
    print("\n---- y_train ----")
    print("\n")
    print(y_train)
    
    print("\n---- y_test ----")
    print("\n")
    print(y_test)

def training_basic_classifier():
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.tree import DecisionTreeClassifier
    
    print("---- Inside training_basic_classifier component ----")
    
    X_train = np.load(f'data/X_train.npy',allow_pickle=True)
    y_train = np.load(f'data/y_train.npy',allow_pickle=True)
    
    model= DecisionTreeClassifier(criterion="entropy")
    model.fit(X_train, y_train)
    
    import pickle
    with open(f'data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\n DecisionTree Classifier is trained on customer data and saved to PV location /data/model.pkl ----")

def predict_on_test_data():
    import pandas as pd
    import numpy as np
    import pickle
    print("---- Inside predict_on_test_data component ----")
    with open(f'data/model.pkl','rb') as f:
        DecisionTree_model = pickle.load(f)
    X_test = np.load(f'data/X_test.npy',allow_pickle=True)
    y_pred = DecisionTree_model.predict(X_test)
    np.save(f'data/y_pred.npy', y_pred)
    
    print("\n---- Predicted classes ----")
    print("\n")
    print(y_pred)

def predict_prob_on_test_data():
    import pandas as pd
    import numpy as np
    import pickle
    print("---- Inside predict_prob_on_test_data component ----")
    with open(f'data/model.pkl','rb') as f:
        DecisionTree_model = pickle.load(f)
    X_test = np.load(f'data/X_test.npy',allow_pickle=True)
    y_pred_prob = DecisionTree_model.predict_proba(X_test)
    np.save(f'data/y_pred_prob.npy', y_pred_prob)
    
    print("\n---- Predicted Probabilities ----")
    print("\n")
    print(y_pred_prob)

def get_metrics():
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    from sklearn import metrics
    print("---- Inside get_metrics component ----")
    y_test = np.load(f'data/y_test.npy',allow_pickle=True)
    y_pred = np.load(f'data/y_pred.npy',allow_pickle=True)
    y_pred_prob = np.load(f'data/y_pred_prob.npy',allow_pickle=True)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    entropy = log_loss(y_test, y_pred_prob)
    
    y_test = np.load(f'data/y_test.npy',allow_pickle=True)
    y_pred = np.load(f'data/y_pred.npy',allow_pickle=True)
    print(metrics.classification_report(y_test, y_pred))
    
    print("\n Model Metrics:", {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)})

create_step_prepare_data = kfp.components.create_component_from_func(
    func=prepare_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0']
)

create_model_building = kfp.components.create_component_from_func(
    func=model_building,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

create_step_train_test_split = kfp.components.create_component_from_func(
    func=train_test_split,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

create_step_training_basic_classifier = kfp.components.create_component_from_func(
    func=training_basic_classifier,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

create_step_predict_on_test_data = kfp.components.create_component_from_func(
    func=predict_on_test_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)
create_step_predict_prob_on_test_data = kfp.components.create_component_from_func(
    func=predict_prob_on_test_data,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

create_step_get_metrics = kfp.components.create_component_from_func(
    func=get_metrics,
    base_image='python:3.7',
    packages_to_install=['pandas==1.2.4','numpy==1.21.0','scikit-learn==0.24.2']
)

@dsl.pipeline(
   name='TEST',
   description='A sample pipeline'
)
# Define parameters to be fed into pipeline
def test(data_path: str):
    vop = dsl.VolumeOp(
    name="t-vol",
    resource_name="t-vol", 
    size="1Gi", 
    modes=dsl.VOLUME_MODE_RWO)
    
    prepare_data_task = create_step_prepare_data().add_pvolumes({data_path: vop.volume})
    model_building = create_model_building().add_pvolumes({data_path: vop.volume}).after(prepare_data_task)
    train_test_split = create_step_train_test_split().add_pvolumes({data_path: vop.volume}).after(model_building)
    classifier_training = create_step_training_basic_classifier().add_pvolumes({data_path: vop.volume}).after(train_test_split)
    log_predicted_class = create_step_predict_on_test_data().add_pvolumes({data_path: vop.volume}).after(classifier_training)
    log_predicted_probabilities = create_step_predict_prob_on_test_data().add_pvolumes({data_path: vop.volume}).after(log_predicted_class)
    log_metrics_task = create_step_get_metrics().add_pvolumes({data_path: vop.volume}).after(log_predicted_probabilities)


    
    
    prepare_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    model_building.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_test_split.execution_options.caching_strategy.max_cache_staleness = "P0D"
    classifier_training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_class.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_probabilities.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_metrics_task.execution_options.caching_strategy.max_cache_staleness = "P0D"


kfp.compiler.Compiler().compile(
    pipeline_func=test,
    package_path='testt1.yaml')    

client = kfp.Client()
test._name_ = 'test'
DATA_PATH = '/data'

import datetime
print(datetime.datetime.now().date())


pipeline_func = test
experiment_name = 'test_exp' +"_"+ str(datetime.datetime.now().date())
run_name = pipeline_func._name_ + ' run'
namespace = "kubeflow"

arguments = {"data_path":DATA_PATH}

kfp.compiler.Compiler().compile(pipeline_func,  
  '{}.zip'.format(experiment_name))

run_result = client.create_run_from_pipeline_func(pipeline_func, 
                                                  experiment_name=experiment_name, 
                                                  run_name=run_name, 
                                                  arguments=arguments)

    
    
    