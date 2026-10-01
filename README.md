# Machine Learning Projects

Two projects for the *Machine Learning* course (MSc in Engineering in Computer Science and AI, Sapienza University of Rome, A.Y. 2025/26), made with **Giacomo Aroni https://github.com/iamgiac**. Each folder has the notebooks and the written report.

| # | Project | Tasks | Stack |
|---|---|---|---|
| 01 | [Diabetes prediction: comparing classifiers](01-diabetes-classification) | Imbalanced binary classification | scikit-learn |
| 02 | [Model performance in reduced feature spaces](02-dimensionality-reduction-cnn-fnn-kmeans) | Image classification, regression, clustering + PCA vs autoencoders | PyTorch, TensorFlow/Keras, scikit-learn |

---

## 01 · Comparative Study of Classification Algorithms for Diabetes Prediction
*November 2025* · [notebook](01-diabetes-classification/diabetes_classification.ipynb) · [report](01-diabetes-classification/report.pdf)

- **Data:** Kaggle *Diabetes Prediction Dataset*, ~100k patients, 8 clinical features, strongly imbalanced target.
- **Pipeline:** noise cleaning, category consolidation (e.g. smoking history), stratified train/val/test split, scaling + one-hot encoding in `Pipeline`s, learning curves.
- **Models:** Naïve Bayes, Decision Tree, Random Forest, Logistic Regression, SVM, tuned with `GridSearchCV` **on F1** and class weighting to favour recall.
- **Results:** Random Forest has the best accuracy (97.1%) and F1 (0.80) but misses many diabetic patients. Logistic Regression and SVM reach **recall 0.90 with ROC AUC 0.97**, the better choice when a false negative is the expensive error.

## 02 · Analyzing Model Performance in Reduced Feature Spaces: CNN, FNN and K-Means
*January 2026* · [report](02-dimensionality-reduction-cnn-fnn-kmeans/report.pdf)

Each task is run on the original features and on 2 compressed representations of the same size (**PCA** vs a **trained autoencoder**) to measure what dimensionality reduction costs or gains.

| Task | Data | Model | Outcome |
|---|---|---|---|
| [Image classification](02-dimensionality-reduction-cnn-fnn-kmeans/1_cnn_fish_classification.ipynb) | Large-Scale Fish Dataset (RGB) | CNN in **PyTorch** | Validation accuracy > 95%, no overfitting gap |
| [Regression](02-dimensionality-reduction-cnn-fnn-kmeans/2_fnn_car_price_regression.ipynb) | Used cars (price) | Feed-forward NN in **Keras** | R² = 0.94 on original features, 0.92 with PCA, 0.91 with autoencoder, at a fraction of the training time |
| [Clustering](02-dimensionality-reduction-cnn-fnn-kmeans/3_kmeans_customer_clustering.ipynb) | Automobile customer segmentation | **K-Means** (elbow + silhouette) | Raw data shows no clear structure (silhouette ≈ 0.23). The autoencoder latent space gives the clearest, most stable clusters (K = 4) |

**Takeaway:** a learned latent space keeps almost all the predictive signal for supervised tasks and *improves* cluster separability for unsupervised ones.

---

### Run
The notebooks download datasets automatically with `kagglehub` (a Kaggle account is required).
```bash
pip install numpy pandas scikit-learn matplotlib mlxtend kagglehub torch torchvision tensorflow
jupyter lab
```
