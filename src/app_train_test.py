from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn
import random
from collections import Counter
import matplotlib
import logging

logging.basicConfig(level=logging.INFO)  # Set the logging level

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define configuration and paths
RESULTS_DIR = 'results'
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
JSON_DIR = os.path.join(RESULTS_DIR, 'json')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Create necessary directories
for directory in [RESULTS_DIR, MODEL_DIR, JSON_DIR, PLOTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Mapping for labels
MAPPING_DICT = {'glitch': 0, 'step': 1, 'drift': 2, 'other': 3, 'nothing': 3}
REVERSE_MAPPING_DICT = {v: k for k, v in MAPPING_DICT.items()}

# Define constants
CENTROID_FILENAME = os.path.join(JSON_DIR, 'clustering_results.json')
MODEL_FILENAME = os.path.join(MODEL_DIR, 'best_LR_model.pkl')

# Features to put aside during training and prediction
FEATURES_TO_EXCLUDE = ['labels', 'max amplitude', 'final amplitude', 'max velocity']


# K-selection agent for determining optimal cluster count
class KSelectionAgent(nn.Module):
    def __init__(self, state_size, min_k=2, max_k=10):
        super(KSelectionAgent, self).__init__()
        self.state_size = state_size
        self.min_k = min_k
        self.max_k = max_k
        self.action_size = max_k - min_k + 1  # The number of `k` choices
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, self.action_size)  # Output probabilities for different `k`

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)  # Output logits for k choices

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(self.min_k, self.max_k)  # Explore different k values
        with torch.no_grad():
            return self.min_k + torch.argmax(self.forward(state)).item()  # Pick best k


def train_agent(data, episodes=100, min_k=2, max_k=10, epsilon=0.1):
    state_size = data.shape[1]  # Number of features (time points)
    agent = KSelectionAgent(state_size, min_k, max_k)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for episode in range(episodes):
        state = torch.tensor(data.mean(axis=0), dtype=torch.float32)  # Use mean time series as state
        k = agent.choose_action(state, epsilon)

        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        labels = kmeans.labels_
        reward = silhouette_score(data, labels)  # Higher silhouette → better clustering

        # Compute loss and update agent
        optimizer.zero_grad()
        output = agent(state)
        target = torch.zeros_like(output)
        target[k - min_k] = reward  # Assign reward to selected k
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    return agent


def get_best_k(agent, data):
    state = torch.tensor(data.mean(axis=0), dtype=torch.float32)
    best_k = agent.choose_action(state, epsilon=0.0)  # Always exploit best learned `k`
    app.logger.info("--- best_k is {}".format(best_k))
    return best_k


def plot_prototypes(data, labels, best_k, filename=None):
    prototypes = np.zeros((best_k, data.shape[1]))
    for i in range(best_k):
        prototypes[i] = data[labels == i].mean(axis=0)  # Compute mean prototype for each cluster

    plt.figure(figsize=(10, 6))
    for i in range(best_k):
        plt.plot(prototypes[i], label=f'Cluster {i}')
    plt.legend()
    plt.xlabel("Time Points")
    plt.ylabel("Value")
    plt.title("Cluster Prototypes")

    if filename:
        plt.savefig(filename)
        plt.close()

    return prototypes


def run_robust_clustering(data, n_runs=100, episodes=100, min_k=4, max_k=4, epsilon=0.1, seed=42):
    """
    Run clustering multiple times to get robust results.
    """
    k_values = []
    all_centroids = []
    all_labels = []
    silhouette_scores = []

    for run in range(n_runs):
        # Set different random seeds for each run
        random_state = run * seed

        # Train agent with different random initialization
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)

        agent = train_agent(data, episodes=episodes, min_k=min_k, max_k=max_k, epsilon=epsilon)
        best_k = get_best_k(agent, data)
        k_values.append(best_k)

        # Run KMeans with the selected k
        kmeans = KMeans(n_clusters=best_k, random_state=random_state).fit(data)
        labels = kmeans.labels_
        all_labels.append(labels)

        # Calculate silhouette score
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)

        # Extract centroids
        centroids = []
        for i in range(best_k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:  # Ensure cluster is not empty
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid.tolist())
        all_centroids.append(centroids)

    # Find most frequent k
    k_counter = Counter(k_values)
    best_k = k_counter.most_common(1)[0][0]

    # Get the run with the highest silhouette score
    best_run_idx = np.argmax(silhouette_scores)
    best_silhouette = silhouette_scores[best_run_idx]
    consensus_centroids = all_centroids[best_run_idx]
    best_labels = all_labels[best_run_idx]

    # Summarize results
    results = {
        "best_k": int(best_k),
        "k_distribution": {str(k): count for k, count in k_counter.items()},
        "consensus_centroids": consensus_centroids,
        "best_silhouette_score": float(best_silhouette),
        "all_silhouette_scores": [float(s) for s in silhouette_scores],
        "reliability": float(k_counter[best_k] / n_runs)  # Proportion of runs that chose the best k
    }

    # Plot the best prototypes
    prototype_plot_path = os.path.join(PLOTS_DIR, 'cluster_prototypes.png')
    best_prototypes = plot_prototypes(data, best_labels, best_k, prototype_plot_path)

    return results, best_prototypes


def save_results_to_json(results, filename=CENTROID_FILENAME):
    """
    Save clustering results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    return filename


def cluster_data_from_json(json_filename, data_df, add_label=False):
    """
    Compute distances to centroids from saved JSON results.
    """
    with open(json_filename, 'r') as f:
        json_data = json.load(f)

    centroids = np.array(json_data['consensus_centroids'])
    distances = euclidean_distances(data_df, centroids)
    cluster_assignments = np.argmin(distances, axis=1)

    result_df = pd.DataFrame(cluster_assignments, columns=['cluster'])

    for i in range(centroids.shape[0]):
        result_df[f'distance_to_centroid_{i}'] = distances[:, i]

    if add_label == False:
        result_df.drop(columns=['cluster'], inplace=True)

    return result_df


def add_external_features(clustered_df, original_df):
    """
    Add back the excluded features to the clustered data.
    """
    # Make a copy to avoid modifying the original
    df_features = clustered_df.copy()

    # Add back the features that were set aside
    for feature in FEATURES_TO_EXCLUDE:
        if feature in original_df.columns:
            df_features[feature] = original_df[feature].values

    return df_features


def add_numeric_labels(df, mapping_dict):
    """
    Convert string labels to numeric labels using the mapping dictionary.
    """
    if 'labels' in df.columns:
        df['num_labels'] = df['labels'].map(mapping_dict)
    return df


def evaluate_predictions(y_true, y_pred, display_labels=None, do_percents=False, save_path=None):
    """
    Evaluate predictions and create confusion matrix visualization.
    """
    accuracy = np.mean(y_true == y_pred)
    confusion_matrix_result = confusion_matrix(y_true, y_pred)

    if do_percents:
        conf_matrix_percent = confusion_matrix_result.astype(float) / confusion_matrix_result.sum(axis=1,
                                                                                                  keepdims=True) * 100
        disp = confusion_matrix(confusion_matrix=conf_matrix_percent, display_labels=display_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=conf_matrix_percent, display_labels=display_labels).plot(
            cmap=plt.cm.Blues, values_format=".2f", ax=ax)

        # Modify text labels to include %
        for text in ax.texts:
            text.set_text(text.get_text() + "%")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=display_labels).plot(
            cmap=plt.cm.Blues, ax=ax)

    plt.title(f"Confusion Matrix - Accuracy {accuracy * 100:.2f}%")

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300)
        plt.close()

    return accuracy, confusion_matrix_result


def train_classifier(X, y, cv_splits=5, param_grid=None):
    """
    Train a logistic regression classifier with optional hyperparameter search.
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    classifier = LogisticRegression(solver='liblinear', random_state=17, max_iter=1000)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=17)
    best_model = None
    best_score = -np.inf

    if param_grid:  # Perform hyperparameter search
        grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', n_jobs=-1,
                                   verbose=1)  # added n_jobs and verbose
        grid_search.fit(X, y)  # Fit on the entire dataset for hyperparameter search
        best_model = grid_search.best_estimator_
        return  grid_search

    # grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    # grid_search.fit(X, y)
    # return grid_search


def get_label_probabilities(model, X):
    """
    Get class probabilities for each prediction.
    """
    return model.predict_proba(X)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train():
    app.logger.info("--- train() start..")
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Load the training dataframe
        train_df = pd.read_csv(file)
        app.logger.info("--- train() found df, start clustering, that will take some time")
        # Extract data-only portion (excluding features to set aside)
        train_data_only = train_df.drop(columns=FEATURES_TO_EXCLUDE, errors='ignore')
        app.logger.info("--- train() train_data_only cols: {}".format(train_data_only.columns))
        # Run clustering to find optimal k and get centroids
        results, _ = run_robust_clustering(
            train_data_only.values,
            n_runs=30,
            episodes=10,
            min_k=4,
            max_k=4,
            epsilon=0.1
        )
        app.logger.info("--- train() finished clustering")
        # Save clustering results
        centroid_file = save_results_to_json(results)
        app.logger.info("--- train() saved centroids")
        # Create features based on distances to centroids
        clustered_df = cluster_data_from_json(centroid_file, train_data_only.values)
        df_features = add_external_features(clustered_df, train_df)
        # Add numeric labels
        df_features = add_numeric_labels(df_features, MAPPING_DICT)
        app.logger.info("--- train() df_features: {}".format(df_features.columns))
        app.logger.info("--- train() df_features dtypes: {}".format(df_features.dtypes))
        # app.logger.info("--- train() df_features number of nans: {}".format(df_features.isna().sum()))

        # Extract features and labels for training
        y = df_features['num_labels'].values
        X = df_features.drop(columns=['num_labels','labels']).values
        app.logger.info("--- train() start classification")
        # Define parameter grid for hyperparameter search
        param_grid_lr = {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        }

        # Train classifier
        app.logger.info("--- train() X shape: {}".format(X.shape))
        model = train_classifier(X, y, 5, param_grid_lr)

        # Save trained model
        joblib.dump(model, MODEL_FILENAME)
        app.logger.info("--- train() model saved")
        # Generate predictions on training set
        y_train_hat = model.predict(X)
        app.logger.info("--- y_train_hat shape: ".format(y_train_hat.shape, y_train_hat.sum()))
        train_accuracy, _ = evaluate_predictions(
            y, y_train_hat,
            display_labels=['Glitch', 'Step', 'Drift', 'O-N'],
            save_path=os.path.join(PLOTS_DIR, 'train_confusion_matrix.pdf')
        )

        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'train_accuracy': float(train_accuracy),
            'best_k': results['best_k'],
            'silhouette_score': results['best_silhouette_score'],
            'model_file': MODEL_FILENAME,
            'centroid_file': centroid_file
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("--- predict() start.")
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Check if model and centroids exist
        if not os.path.exists(MODEL_FILENAME) or not os.path.exists(CENTROID_FILENAME):
            return jsonify({'error': 'Model or centroids not found. Please train first.'}), 400

        # Load the test dataframe
        test_df = pd.read_csv(file)
        app.logger.info("--- predict() test_df cols: ".format(test_df.columns))
        has_labels = 'labels' in test_df.columns
        app.logger.info("--- predict() got data")
        # Extract data-only portion (excluding features to set aside)
        test_data_only = test_df.drop(columns=FEATURES_TO_EXCLUDE, errors='ignore')
        app.logger.info("--- predict() test_data_only cols: ".format(test_data_only.columns))
        # Load model
        model = joblib.load(MODEL_FILENAME)

        # Create features based on distances to centroids
        clustered_df = cluster_data_from_json(CENTROID_FILENAME, test_data_only.values)

        # Add back the excluded features
        df_features = add_external_features(clustered_df, test_df)
        app.logger.info("--- predict() df_features shape: ".format(df_features.shape))
        # Prepare for prediction
        if has_labels:
            df_features = add_numeric_labels(df_features, MAPPING_DICT)
            y_test = df_features['num_labels'].values
            X_test = df_features.drop(columns=['num_labels','labels']).values
        else:
            X_test = df_features.values

        # Generate predictions
        y_pred = model.predict(X_test)

        # Get probabilities
        probabilities = get_label_probabilities(model, X_test)

        # Convert numeric predictions back to string labels
        pred_labels = [REVERSE_MAPPING_DICT[pred] for pred in y_pred]

        # Create result DataFrame
        result_df = pd.DataFrame({
            'predicted_label': pred_labels,
            'predicted_class': y_pred.tolist()
        })

        # Add probabilities for each class
        for i, class_name in REVERSE_MAPPING_DICT.items():
            result_df[f'prob_{class_name}'] = probabilities[:, i]

        # Calculate and report accuracy if labels are available
        if has_labels:
            accuracy, _ = evaluate_predictions(
                y_test, y_pred,
                display_labels=['Glitch', 'Step', 'Drift', 'O-N'],
                save_path=os.path.join(PLOTS_DIR, 'test_confusion_matrix.pdf')
            )
            result_df['actual_label'] = [REVERSE_MAPPING_DICT[y] for y in y_test]
            result_df['correct'] = (y_test == y_pred)

            response = {
                'success': True,
                'predictions': result_df.to_dict(orient='records'),
                'accuracy': float(accuracy)
            }
        else:
            response = {
                'success': True,
                'predictions': result_df.to_dict(orient='records')
            }

        # Save predictions
        from datetime import datetime
        now_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        result_df.to_csv(os.path.join(RESULTS_DIR, f'predictions_{now_date}.csv'), index=False)


        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/visualize_clusters', methods=['GET'])
def visualize_clusters():
    if not os.path.exists(os.path.join(PLOTS_DIR, 'cluster_prototypes.png')):
        return jsonify({'error': 'Cluster visualization not available. Train the model first.'}), 400

    return jsonify({
        'success': True,
        'image_path': os.path.join(PLOTS_DIR, 'cluster_prototypes.png')
    })


if __name__ == '__main__':
    # Create a simple HTML template for the frontend
    if not os.path.exists('templates'):
        os.makedirs('templates')

    with open('templates/index.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Time Series Classifier</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 800px; margin: 0 auto; }
                .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                h2 { color: #333; }
                .results { margin-top: 20px; padding: 10px; background-color: #f8f8f8; }
                .hidden { display: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Time Series Classifier</h1>

                <div class="section">
                    <h2>Train Model</h2>
                    <form id="trainForm" enctype="multipart/form-data">
                        <div>
                            <label for="trainFile">Upload Training CSV:</label>
                            <input type="file" id="trainFile" name="file" accept=".csv">
                        </div>
                        <div style="margin-top: 10px;">
                            <button type="button" onclick="trainModel()">Train Model</button>
                        </div>
                    </form>
                    <div id="trainResults" class="results hidden"></div>
                </div>

                <div class="section">
                    <h2>Make Predictions</h2>
                    <form id="predictForm" enctype="multipart/form-data">
                        <div>
                            <label for="testFile">Upload Test CSV:</label>
                            <input type="file" id="testFile" name="file" accept=".csv">
                        </div>
                        <div style="margin-top: 10px;">
                            <button type="button" onclick="makePredictions()">Predict</button>
                        </div>
                    </form>
                    <div id="predictResults" class="results hidden"></div>
                </div>

                <div class="section">
                    <h2>Visualizations</h2>
                    <button type="button" onclick="visualizeClusters()">View Cluster Prototypes</button>
                    <div id="visualResults" class="results hidden">
                        <img id="clusterImage" style="max-width: 100%;">
                    </div>
                </div>
            </div>

            <script>
                function trainModel() {
                    const form = document.getElementById('trainForm');
                    const formData = new FormData(form);
                    const resultsDiv = document.getElementById('trainResults');

                    resultsDiv.innerHTML = 'Training in progress...';
                    resultsDiv.classList.remove('hidden');

                    fetch('/train', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                        } else {
                            resultsDiv.innerHTML = `
                                <p>Model trained successfully!</p>
                                <p>Training Accuracy: ${(data.train_accuracy * 100).toFixed(2)}%</p>
                                <p>Best K: ${data.best_k}</p>
                                <p>Silhouette Score: ${data.silhouette_score.toFixed(4)}</p>
                            `;
                        }
                    })
                    .catch(error => {
                        resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                    });
                }

                function makePredictions() {
                    const form = document.getElementById('predictForm');
                    const formData = new FormData(form);
                    const resultsDiv = document.getElementById('predictResults');

                    resultsDiv.innerHTML = 'Generating predictions...';
                    resultsDiv.classList.remove('hidden');

                    fetch('/predict', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                        } else {
                            let resultsHtml = '<p>Predictions generated successfully!</p>';

                            if (data.accuracy !== undefined) {
                                resultsHtml += `<p>Test Accuracy: ${(data.accuracy * 100).toFixed(2)}%</p>`;
                            }

                            resultsHtml += '<p>Results saved to predictions.csv</p>';

                            if (data.predictions && data.predictions.length > 0) {
                                resultsHtml += '<p>First 5 predictions:</p><table border="1" style="border-collapse: collapse; width: 100%;">';

                                // Create header row based on keys of first prediction
                                const headers = Object.keys(data.predictions[0]);
                                resultsHtml += '<tr>';
                                headers.forEach(header => {
                                    resultsHtml += `<th style="padding: 5px;">${header}</th>`;
                                });
                                resultsHtml += '</tr>';

                                // Add data rows (first 5 only)
                                const displayPredictions = data.predictions.slice(0, 5);
                                displayPredictions.forEach(pred => {
                                    resultsHtml += '<tr>';
                                    headers.forEach(header => {
                                        let value = pred[header];
                                        if (typeof value === 'number') {
                                            value = value.toFixed(4);
                                        }
                                        resultsHtml += `<td style="padding: 5px;">${value}</td>`;
                                    });
                                    resultsHtml += '</tr>';
                                });

                                resultsHtml += '</table>';
                            }

                            resultsDiv.innerHTML = resultsHtml;
                        }
                    })
                    .catch(error => {
                        resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                    });
                }

                function visualizeClusters() {
                    const resultsDiv = document.getElementById('visualResults');
                    const clusterImage = document.getElementById('clusterImage');

                    fetch('/visualize_clusters')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                            resultsDiv.classList.remove('hidden');
                        } else {
                            clusterImage.src = data.image_path + '?t=' + new Date().getTime();  // Add timestamp to prevent caching
                            resultsDiv.classList.remove('hidden');
                        }
                    })
                    .catch(error => {
                        resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                        resultsDiv.classList.remove('hidden');
                    });
                }
            </script>
        </body>
        </html>
        ''')

    app.run(debug=True, host='0.0.0.0', port=5000)
