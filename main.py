# This is an Ayurvedic prakriti analysis system
import pandas as pd
import numpy as np
import os
import zipfile
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class PrakritiDoshaModel:
    def __init__(self):
        self.dt_model = None
        self.rf_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the prakriti dosha dataset"""
        try:
         
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(file_path, encoding=encoding)
                    print(f"✓ Dataset loaded successfully with {encoding} encoding!")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("Could not load file with any encoding")
                return None
            
            print(f"Dataset shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            print("\nDataset Info:")
            print(self.data.info())
            
            print("\nFirst few rows:")
            print(self.data.head())
            
            print("\nMissing values:")
            missing = self.data.isnull().sum()
            if missing.sum() > 0:
                print(missing[missing > 0])
            else:
                print("No missing values found!")
            
            # Display unique values for each column (first 10)
            print("\nUnique values per column (first 10):")
            for col in self.data.columns:
                unique_vals = self.data[col].unique()
                print(f"{col}: {unique_vals[:10]}")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def encode_categorical_features(self, X, y=None, fit=True):
        """Encode categorical features using LabelEncoder"""
        X_encoded = X.copy()
        
        for column in X_encoded.columns:
            if X_encoded[column].dtype == 'object':
                if fit:
                    self.label_encoders[column] = LabelEncoder()
                    X_encoded[column] = self.label_encoders[column].fit_transform(X_encoded[column].astype(str))
                else:
                    if column in self.label_encoders:
                        X_encoded[column] = self.label_encoders[column].transform(X_encoded[column].astype(str))
        
        if y is not None and fit:
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y_encoded = self.target_encoder.fit_transform(y.astype(str))
                self.target_names = self.target_encoder.classes_
                return X_encoded, y_encoded
            return X_encoded, y
        
        return X_encoded
    
    def prepare_data(self, target_column, test_size=0.2, random_state=42):
        """Prepare data for training"""
        if self.data is None:
            print("Please load dataset first!")
            return None
        
        # Check if target column exists
        if target_column not in self.data.columns:
            print(f"Target column '{target_column}' not found!")
            print(f"Available columns: {list(self.data.columns)}")
            return None
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        self.feature_names = list(X.columns)
        
        X_encoded, y_encoded = self.encode_categorical_features(X, y, fit=True)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print(f"✓ Data prepared successfully!")
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        print(f"Number of features: {self.X_train.shape[1]}")
        
        if hasattr(self, 'target_names'):
            print(f"Target classes: {self.target_names}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_decision_tree(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """Train Decision Tree model"""
        print("Training Decision Tree model...")
        
        self.dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        self.dt_model.fit(self.X_train, self.y_train)
        
        dt_train_pred = self.dt_model.predict(self.X_train)
        dt_test_pred = self.dt_model.predict(self.X_test)
        
        dt_train_acc = accuracy_score(self.y_train, dt_train_pred)
        dt_test_acc = accuracy_score(self.y_test, dt_test_pred)
        
        print(f"✓ Decision Tree trained!")
        print(f"  Training Accuracy: {dt_train_acc:.4f}")
        print(f"  Testing Accuracy: {dt_test_acc:.4f}")
        
        return self.dt_model
    
    def train_random_forest(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        
        rf_train_pred = self.rf_model.predict(self.X_train)
        rf_test_pred = self.rf_model.predict(self.X_test)
        
        rf_train_acc = accuracy_score(self.y_train, rf_train_pred)
        rf_test_acc = accuracy_score(self.y_test, rf_test_pred)
        
        print(f"✓ Random Forest trained!")
        print(f"  Training Accuracy: {rf_train_acc:.4f}")
        print(f"  Testing Accuracy: {rf_test_acc:.4f}")
        
        return self.rf_model
    
    def evaluate_models(self):
        """Evaluate both models and display results"""
        if self.dt_model is None or self.rf_model is None:
            print("Please train both models first!")
            return
        
        dt_pred = self.dt_model.predict(self.X_test)
        rf_pred = self.rf_model.predict(self.X_test)
        
        print("="*60)
        print("                MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Decision Tree Results
        print("\n DECISION TREE RESULTS:")
        print("-" * 40)
        dt_acc = accuracy_score(self.y_test, dt_pred)
        print(f"Accuracy: {dt_acc:.4f}")
        
        print("\nClassification Report:")
        if hasattr(self, 'target_names'):
            print(classification_report(self.y_test, dt_pred, target_names=self.target_names))
        else:
            print(classification_report(self.y_test, dt_pred))
        
        print("\nConfusion Matrix:")
        dt_cm = confusion_matrix(self.y_test, dt_pred)
        print(dt_cm)
        
        # Random Forest Results
        print("\n RANDOM FOREST RESULTS:")
        print("-" * 40)
        rf_acc = accuracy_score(self.y_test, rf_pred)
        print(f"Accuracy: {rf_acc:.4f}")
        
        print("\nClassification Report:")
        if hasattr(self, 'target_names'):
            print(classification_report(self.y_test, rf_pred, target_names=self.target_names))
        else:
            print(classification_report(self.y_test, rf_pred))
        
        print("\nConfusion Matrix:")
        rf_cm = confusion_matrix(self.y_test, rf_pred)
        print(rf_cm)
        
        # Cross-validation scores
        print("\n CROSS-VALIDATION SCORES:")
        print("-" * 40)
        try:
            dt_cv_scores = cross_val_score(self.dt_model, self.X_train, self.y_train, cv=5)
            rf_cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5)
            
            print(f"Decision Tree CV: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std() * 2:.4f})")
            print(f"Random Forest CV: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
        except Exception as e:
            print(f"Cross-validation error: {e}")
        
        # Model comparison
        print(f"\n WINNER: {'Random Forest' if rf_acc > dt_acc else 'Decision Tree'}")
        print(f"Accuracy difference: {abs(rf_acc - dt_acc):.4f}")
    
    def show_feature_importance(self):
        """Display feature importance in text format"""
        if self.dt_model is None or self.rf_model is None:
            print("Please train both models first!")
            return
        
        print("\n FEATURE IMPORTANCE:")
        print("="*50)
        
        # Decision Tree
        print("\n Decision Tree - Top 10 Important Features:")
        dt_importance = self.dt_model.feature_importances_
        dt_features = [(self.feature_names[i], dt_importance[i]) for i in range(len(dt_importance))]
        dt_features.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(dt_features[:10]):
            print(f"{i+1:2d}. {feature:<20} : {importance:.4f}")
        
        # Random Forest
        print("\n Random Forest - Top 10 Important Features:")
        rf_importance = self.rf_model.feature_importances_
        rf_features = [(self.feature_names[i], rf_importance[i]) for i in range(len(rf_importance))]
        rf_features.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(rf_features[:10]):
            print(f"{i+1:2d}. {feature:<20} : {importance:.4f}")
    
    def predict_dosha(self, input_data, model_type='rf'):
        """Predict dosha for new input data"""
        if model_type == 'dt' and self.dt_model is None:
            print("Decision Tree model not trained!")
            return None
        elif model_type == 'rf' and self.rf_model is None:
            print("Random Forest model not trained!")
            return None
        
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        input_encoded = self.encode_categorical_features(input_df, fit=False)
        
        if model_type == 'dt':
            prediction = self.dt_model.predict(input_encoded)
            probability = self.dt_model.predict_proba(input_encoded)
        else:
            prediction = self.rf_model.predict(input_encoded)
            probability = self.rf_model.predict_proba(input_encoded)
        
        if hasattr(self, 'target_encoder'):
            prediction_label = self.target_encoder.inverse_transform(prediction)
        else:
            prediction_label = prediction
        
        return prediction_label, probability

def main_workflow():
    """Complete workflow for Prakriti Dosha detection"""
    print("🕉  PRAKRITI DOSHA DETECTION SYSTEM 🕉️")
    print("="*50)
    
    # Initialize model
    dosha_model = PrakritiDoshaModel()
    
    # Step 1: Load dataset
    print("\n STEP 1: Loading Dataset")
    print("-" * 30)
    
   
    possible_files = [    
        r'C:\ProgramData\extracted_data\Prakriti.csv',        
    ]
    
    dataset_path = None
    
    if os.path.exists('dataset_path.txt'):
        with open('dataset_path.txt', 'r') as f:
            dataset_path = f.read().strip()
        print(f"Using saved path: {dataset_path}")
    else:
       
        for path in possible_files[1:]:
            if os.path.exists(path):
                dataset_path = path
                print(f"Found dataset: {dataset_path}")
                break
    
    if not dataset_path:
        print(" No dataset found! Please:")
        print("1. First extract your ZIP file using the extraction code")
        print("2. The ZIP file should be at: C:\\ProgramData\\Ayurveda prakriti dosh.zip")
        print("3. It contains: Prakriti.csv (602 KB) and data.csv (51 KB)")
        
       
        print("\n Attempting to extract ZIP file automatically...")
        zip_path = r"C:\ProgramData\Ayurveda prakriti dosh.zip"
        if os.path.exists(zip_path):
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("./extracted_data/")
                print(" ZIP extracted successfully!")
                
             
                main_dataset = "./extracted_data/Prakriti.csv"
                if os.path.exists(main_dataset):
                    dataset_path = main_dataset
                    print(f" Found main dataset: {dataset_path}")
                else:
                    alt_dataset = "./extracted_data/data.csv"
                    if os.path.exists(alt_dataset):
                        dataset_path = alt_dataset
                        print(f" Found alternative dataset: {dataset_path}")
            except Exception as e:
                print(f" Extraction failed: {e}")
                return None
        else:
            print(f" ZIP file not found at: {zip_path}")
            return None
    
    if not dataset_path:
        print(" Could not locate dataset file")
        return None
    
    data = dosha_model.load_and_preprocess_data(dataset_path)
    if data is None:
        return None
    
    # Step 2: Prepare data
    print(f"\n STEP 2: Preparing Data")
    print("-" * 30)
    
    # Auto-detect target column
    target_column = None
    target_keywords = ['dosha', 'prakriti', 'constitution', 'type', 'class', 'label', 'target']
    
    for col in data.columns:
        if any(keyword in col.lower() for keyword in target_keywords):
            target_column = col
            print(f"Auto-detected target column: '{target_column}'")
            break
    
    if not target_column:
        print("Available columns:")
        for i, col in enumerate(data.columns):
            print(f"{i+1}. {col}")
        choice = input("Enter target column name or number: ")
        if choice.isdigit():
            target_column = data.columns[int(choice)-1]
        else:
            target_column = choice
    
    result = dosha_model.prepare_data(target_column)
    if result is None:
        return None
    
    # Step 3: Train models
    print(f"\n STEP 3: Training Models")
    print("-" * 30)
    
    dt_model = dosha_model.train_decision_tree()
    rf_model = dosha_model.train_random_forest()
    
    # Step 4: Evaluate models
    print(f"\n📊 STEP 4: Evaluating Models")
    print("-" * 30)
    
    dosha_model.evaluate_models()
    
    # Step 5: feature importance
    dosha_model.show_feature_importance()
    
    # Step 6: Example prediction
    print(f"\n STEP 5: Example Prediction")
    print("-" * 30)
    
    print("Model is ready for predictions!")
    print("Use dosha_model.predict_dosha(sample_data) to make predictions")
    
    return dosha_model

if __name__ == "__main__":
    model = main_workflow()