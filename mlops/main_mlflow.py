import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif
import os
import pickle
import json
from woe import WoE_Binning
import mlflow
import mlflow.sklearn

# Load dữ liệu
def load_data(data_path='D:/DH/Thuc Tap/pj/data/'):
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    dataframes = [pd.read_csv(os.path.join(data_path, file), low_memory=False) for file in csv_files]
    return pd.concat(dataframes, ignore_index=True), csv_files

# Hàm lưu kết quả của một phiên bản mô hình
def save_model_results(version, model, metrics, summary_table, df_scorecard, threshold, csv_files):
    version_dir = f'D:/DH/Thuc Tap/pj/mlops/{version}/'
    backend_dir = 'D:/DH/Thuc Tap/pj/webapp/backend/'
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(backend_dir, exist_ok=True)

    # Lưu model
    with open(os.path.join(version_dir, 'pipeline_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(backend_dir, 'pipeline_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # Chuyển đổi các giá trị trong metrics sang kiểu dữ liệu Python tiêu chuẩn
    metrics_to_save = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.number))}

    # Lưu các metrics vào file JSON (chỉ lưu các chỉ số chính)
    with open(os.path.join(version_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=4)

    # Lưu feature score
    summary_table.to_csv(os.path.join(version_dir, 'feature_score.csv'), index=False)

    # Lưu scorecard_scores.csv
    scorecard_scores_series = df_scorecard['Score - Preliminary']
    scorecard_scores_series.to_csv(os.path.join(version_dir, 'scorecard_scores.csv'), index=False)
    scorecard_scores_series.to_csv(os.path.join(backend_dir, 'scorecard_scores.csv'), index=False)

    # Lưu threshold
    with open(os.path.join(version_dir, 'threshold.txt'), 'w') as f:
        f.write(str(threshold))
    with open(os.path.join(backend_dir, 'threshold.txt'), 'w') as f:
        f.write(str(threshold))

    # Lưu danh sách các file CSV đã sử dụng
    with open(os.path.join(version_dir, 'data_files.txt'), 'w') as f:
        for file in csv_files:
            f.write(f"{file}\n")

    # Vẽ và lưu các biểu đồ
    save_confusion_matrix(np.array(metrics['confusion_matrix']), os.path.join(version_dir, 'confusion_matrix.png'))
    save_roc_curve(metrics['roc_curve'], os.path.join(version_dir, 'roc_curve.png'))
    save_youden_plot(metrics['youden_statistic'], threshold, os.path.join(version_dir, 'youden_statistic_plot.png'))

    # Log các metrics vào MLflow
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1", metrics["f1"])
    mlflow.log_metric("auc", metrics["auc"])
    mlflow.log_metric("gini", metrics["gini"])

    # Log các artifacts vào MLflow
    mlflow.log_artifact(os.path.join(version_dir, 'pipeline_model.pkl'))
    mlflow.log_artifact(os.path.join(version_dir, 'feature_score.csv'))
    mlflow.log_artifact(os.path.join(version_dir, 'scorecard_scores.csv'))
    mlflow.log_artifact(os.path.join(version_dir, 'threshold.txt'))
    mlflow.log_artifact(os.path.join(version_dir, 'confusion_matrix.png'))
    mlflow.log_artifact(os.path.join(version_dir, 'roc_curve.png'))
    mlflow.log_artifact(os.path.join(version_dir, 'youden_statistic_plot.png'))

# Hàm vẽ và lưu Confusion Matrix
def save_confusion_matrix(cm, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(path)
    plt.close()

# Hàm vẽ và lưu ROC Curve
def save_roc_curve(roc_data, path):
    fpr, tpr, _ = roc_data
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(path)
    plt.close()

# Hàm vẽ và lưu Youden's J-Statistic plot
def save_youden_plot(j_statistic, threshold, path):
    thresholds = j_statistic['thresholds']
    J = j_statistic['J']
    best_index = j_statistic['best_index']
    best_thresh = thresholds[best_index]

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, J, color='blue', linestyle='-', linewidth=2, label='Youden\'s J-Statistic')
    plt.scatter(best_thresh, J[best_index], color='red', s=100, label='Best Threshold')  # Đánh dấu ngưỡng tốt nhất
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Youden's J-Statistic", fontsize=14)
    plt.title("Youden's J-Statistic vs. Threshold", fontsize=16)
    plt.legend(loc='best')

    # Thêm chú thích cho ngưỡng tốt nhất
    plt.annotate(f'Best Threshold: {best_thresh:.2f}', 
                 xy=(best_thresh, J[best_index]), 
                 xytext=(best_thresh + 0.1, J[best_index] + 0.1),
                 arrowprops=dict(facecolor='green', arrowstyle='->'),
                 fontsize=12,
                 color='darkgreen')

    # Thêm lưới cho biểu đồ
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Điều chỉnh giới hạn trục
    plt.xlim([0.0, 1.05])
    plt.ylim([min(J) - 0.05, max(J) + 0.05])

    # Lưu biểu đồ vào đường dẫn
    plt.savefig(path)
    plt.close()

# Hàm tính toán các chỉ số của mô hình
def calculate_metrics(model, X_test, y_test, X_train, y_train):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    gini = auc * 2 - 1

    # Youden's J-Statistic
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, model.predict_proba(X_train)[:, 1], drop_intermediate=False)
    J = tpr_train - fpr_train
    best_index = np.argmax(J)
    best_threshold = thresholds_train[best_index]

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'gini': gini,
        'confusion_matrix': cm.tolist(),
        'roc_curve': (fpr.tolist(), tpr.tolist(), thresholds.tolist()),
        'youden_statistic': {'J': J.tolist(), 'thresholds': thresholds_train.tolist(), 'best_index': best_index}
    }

    return metrics, best_threshold

# Hàm huấn luyện mô hình
def train_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Loan Default Prediction")

    with mlflow.start_run():
        # Load dữ liệu
        data, csv_files = load_data()
        
        # Tiền xử lý dữ liệu (như đã liệt kê chi tiết trước đó)
        loan_data = data.copy()
        loan_data['target'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']), 0, 1)
        loan_data.drop(columns=['loan_status'], inplace=True)
        loan_data.drop(columns=['id', 'member_id', 'policy_code', 'total_rec_prncp', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee'], inplace=True)
        loan_data.drop(columns=['emp_title', 'url', 'desc', 'title', 'zip_code', 'addr_state', 'pymnt_plan', 'sub_grade', 'next_pymnt_d'], inplace=True)

        def loan_term_converter(df, column):
            df[column] = pd.to_numeric(df[column].str.replace(r' months', ''))
        loan_term_converter(loan_data, 'term')

        def emp_length_converter(df, column):
            df[column] = df[column].str.replace(r'\+ years', '')
            df[column] = df[column].str.replace('< 1 year', '0')
            df[column] = df[column].str.replace(' years', '')
            df[column] = df[column].str.replace(' year', '')
            df[column] = df[column].str.replace(r'\+', '', regex=True)
            df[column] = pd.to_numeric(df[column], errors='coerce')
        emp_length_converter(loan_data, 'emp_length')

        loan_data['earliest_cr_line'] = pd.to_datetime(loan_data['earliest_cr_line'], format='%b-%y')
        loan_data['issue_d'] = pd.to_datetime(loan_data['issue_d'], format='%b-%y')
        loan_data['last_pymnt_d'] = pd.to_datetime(loan_data['last_pymnt_d'], format='%b-%y')
        loan_data['last_credit_pull_d'] = pd.to_datetime(loan_data['last_credit_pull_d'], format='%b-%y')

        # Chia train/test
        X = loan_data.drop('target', axis=1)
        y = loan_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        def calculate_month_difference(date_series, reference_date):
            year_diff = reference_date.year - date_series.dt.year
            month_diff = reference_date.month - date_series.dt.month
            total_month_diff = year_diff * 12 + month_diff
            return total_month_diff

        def date_columns(df, column):
            today_date = pd.to_datetime('2020-08-01')
            df[column] = pd.to_datetime(df[column], format="%b-%y")
            df['mths_since_' + column] = calculate_month_difference(df[column], today_date)
            df['mths_since_' + column] = df['mths_since_' + column].apply(lambda x: df['mths_since_' + column].max() if x < 0 else x)
            df.drop(columns=[column], inplace=True)

        date_columns(X_train, 'earliest_cr_line')
        date_columns(X_train, 'issue_d')
        date_columns(X_train, 'last_pymnt_d')
        date_columns(X_train, 'last_credit_pull_d')

        date_columns(X_test, 'earliest_cr_line')
        date_columns(X_test, 'issue_d')
        date_columns(X_test, 'last_pymnt_d')
        date_columns(X_test, 'last_credit_pull_d')

        # Xử lý missing values
        cols_before = set(X_train.columns)
        X_train.dropna(thresh=X_train.shape[0]*0.60, axis=1, inplace=True)
        cols_after = set(X_train.columns)
        cols_dropped = list(cols_before - cols_after)
        X_test.drop(columns=cols_dropped, inplace=True)

        # Loại bỏ những trường không cần thiết
        cat_features = X_train.select_dtypes(include=['object'])
        chi2_check = {}

        for column in cat_features:
            chi, p, dof, ex = chi2_contingency(pd.crosstab(y_train, cat_features[column]))
            chi2_check.setdefault('Feature', []).append(column)
            chi2_check.setdefault('p-value', []).append(round(p, 10))

        chi2_result = pd.DataFrame(data=chi2_check)
        chi2_result.sort_values(by=['p-value'], ascending=True, ignore_index=True, inplace=True)

        X_train_num = X_train.select_dtypes(include='number').copy()
        X_train_num.fillna(X_train_num.mean(), inplace=True)

        F_statistic, p_values = f_classif(X_train_num, y_train)
        ANOVA_F_table = pd.DataFrame(data={'Numerical_Feature': X_train_num.columns.values, 'F-Score': F_statistic, 'p values': p_values.round(decimals=10)})
        ANOVA_F_table.sort_values(by=['F-Score'], ascending=False, ignore_index=True, inplace=True)

        top_num_features = ANOVA_F_table.iloc[:21, 0].to_list()

        drop_columns_list = ANOVA_F_table.iloc[21:, 0].to_list()
        drop_columns_list.extend(chi2_result.iloc[4:, 0].to_list())
        drop_columns_list.extend(['out_prncp_inv', 'total_pymnt_inv', 'installment', 'revol_bal'])

        def col_to_drop(df, columns_list):
            df.drop(columns=columns_list, inplace=True)

        col_to_drop(X_train, drop_columns_list)
        col_to_drop(X_test, drop_columns_list)

        # Tạo dummy variables
        def dummy_creation(df, columns_list):
            df_dummies = []
            for col in columns_list:
                df_dummies.append(pd.get_dummies(df[col], prefix=col, prefix_sep=':'))
            df_dummies = pd.concat(df_dummies, axis=1)
            df = pd.concat([df, df_dummies], axis=1)
            return df

        X_train = dummy_creation(X_train, ['grade', 'home_ownership', 'verification_status', 'purpose'])
        X_test = dummy_creation(X_test, ['grade', 'home_ownership', 'verification_status', 'purpose'])
        X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)

        # Định nghĩa các biến cho model
        ref_categories = ['grade:G',  'home_ownership:MORTGAGE', 'verification_status:Not Verified', 'purpose:major_purch__car__home_impr', 'term:60', 'int_rate:>20.281', 
                          'annual_inc:>150K', 'dti:>35.191', 'inq_last_6mths:>4',  'revol_util:>1.0', 'out_prncp:>15,437', 'total_pymnt:>25,000', 'total_rec_int:>7,260', 
                          'total_rev_hi_lim:>79,780', 'mths_since_earliest_cr_line:>434',  'mths_since_issue_d:>122', 'mths_since_last_credit_pull_d:>75']

        # Định nghĩa pipeline và train model
        reg = LogisticRegression(max_iter=1000, class_weight='balanced')
        woe_transform = WoE_Binning(X_train)
        pipeline = Pipeline(steps=[('woe', woe_transform), ('model', reg)])
        pipeline.fit(X_train, y_train)

        # Tạo bảng feature score
        X_train_woe_transformed = pipeline.named_steps['woe'].fit_transform(X_train)
        feature_name = X_train_woe_transformed.columns.values
        summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
        summary_table['Coefficients'] = np.transpose(pipeline['model'].coef_)
        summary_table.index = summary_table.index + 1
        summary_table.loc[0] = ['Intercept', pipeline['model'].intercept_[0]]
        summary_table.sort_index(inplace=True)

        # Tạo dataframe cho ref_categories và tính toán min/max coefficients
        df_ref_categories = pd.DataFrame(ref_categories, columns=['Feature name'])
        df_ref_categories['Coefficients'] = 0
        df_scorecard = pd.concat([summary_table, df_ref_categories])
        df_scorecard.reset_index(inplace=True)
        df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

        # Tính toán min/max coefficients
        min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
        max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()

        # Tính điểm số cho từng biến
        df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (850 - 300) / (max_sum_coef - min_sum_coef)
        df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard.loc[0, 'Coefficients'] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (850 - 300) + 300
        df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()

        # Tính các chỉ số và lưu kết quả
        metrics, best_threshold = calculate_metrics(pipeline, X_test, y_test, X_train, y_train)

        # Tính điểm ngưỡng tín dụng và lưu vào threshold.txt
        fpr, tpr, thresholds = roc_curve(y_train, pipeline.predict_proba(X_train)[:, 1], drop_intermediate=False)
        J = tpr - fpr
        max_index = np.argmax(J)
        best_thresh = thresholds[max_index]
        df_cutoffs = pd.DataFrame(thresholds, columns=['thresholds'])
        df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coef) * 
                               ((850 - 300) / (max_sum_coef - min_sum_coef)) + 300).round()
        filtered_cutoffs = df_cutoffs[(df_cutoffs['thresholds'] == best_thresh)]
        final_threshold = filtered_cutoffs['Score'].values[0]

        # Xác định phiên bản mô hình
        model_version = len([d for d in os.listdir('D:/DH/Thuc Tap/pj/mlops') if os.path.isdir(os.path.join('D:/DH/Thuc Tap/pj/mlops', d)) and d.startswith('Model_v')]) + 1
        model_version_name = f'Model_v{model_version}'

        # Lưu kết quả
        save_model_results(model_version_name, pipeline, metrics, summary_table, df_scorecard, final_threshold, csv_files)

        ################# random search #####################
        # # Cấu hình tham số cho Random Search
        # param_distributions = {
        #     'model__C': [0.1, 1, 10],  # Chọn 3 giá trị cố định cho C
        #     'model__solver': ['lbfgs', 'liblinear']  # 2 giá trị cho solver
        # }

        # # Khởi tạo mô hình LogisticRegression
        # reg = LogisticRegression(max_iter=1000, class_weight='balanced')

        # # Tạo pipeline
        # woe_transform = WoE_Binning(X_train)
        # pipeline = Pipeline(steps=[('woe', woe_transform), ('model', reg)])

        # # Khởi tạo RandomizedSearchCV
        # random_search = RandomizedSearchCV(
        #     pipeline, 
        #     param_distributions=param_distributions, 
        #     n_iter=3,  # Thử nghiệm với 3 tổ hợp tham số
        #     cv=2,  # Sử dụng 2-fold cross-validation
        #     scoring='roc_auc',  # Đánh giá theo ROC AUC
        #     verbose=1,  # Hiển thị tiến trình
        #     random_state=42,  # Đảm bảo kết quả có thể tái tạo
        #     n_jobs=-1  # Sử dụng tất cả các nhân CPU có sẵn
        # )

        # # Huấn luyện mô hình
        # random_search.fit(X_train, y_train)

        # best_pipeline = random_search.best_estimator_

        # # Tạo bảng feature score
        # X_train_woe_transformed = best_pipeline.named_steps['woe'].fit_transform(X_train)
        # feature_name = X_train_woe_transformed.columns.values
        # summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
        # summary_table['Coefficients'] = np.transpose(best_pipeline['model'].coef_)
        # summary_table.index = summary_table.index + 1
        # summary_table.loc[0] = ['Intercept', best_pipeline['model'].intercept_[0]]
        # summary_table.sort_index(inplace=True)

        # # Tạo dataframe cho ref_categories và tính toán min/max coefficients
        # df_ref_categories = pd.DataFrame(ref_categories, columns=['Feature name'])
        # df_ref_categories['Coefficients'] = 0
        # df_scorecard = pd.concat([summary_table, df_ref_categories])
        # df_scorecard.reset_index(inplace=True)
        # df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

        # # Tính toán min/max coefficients
        # min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
        # max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()

        # # Tính điểm số cho từng biến
        # df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (850 - 300) / (max_sum_coef - min_sum_coef)
        # df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard.loc[0, 'Coefficients'] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (850 - 300) + 300
        # df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()

        # # Tính các chỉ số và lưu kết quả
        # metrics, best_threshold = calculate_metrics(best_pipeline, X_test, y_test, X_train, y_train)

        # # Tính điểm ngưỡng tín dụng và lưu vào threshold.txt
        # fpr, tpr, thresholds = roc_curve(y_train, best_pipeline.predict_proba(X_train)[:, 1], drop_intermediate=False)
        # J = tpr - fpr
        # max_index = np.argmax(J)
        # best_thresh = thresholds[max_index]
        # df_cutoffs = pd.DataFrame(thresholds, columns=['thresholds'])
        # df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coef) * 
        #                     ((850 - 300) / (max_sum_coef - min_sum_coef)) + 300).round()
        # filtered_cutoffs = df_cutoffs[(df_cutoffs['thresholds'] == best_thresh)]
        # final_threshold = filtered_cutoffs['Score'].values[0]

        # # Xác định phiên bản mô hình
        # model_version = len([d for d in os.listdir('D:/DH/Thuc Tap/pj/mlops') if os.path.isdir(os.path.join('D:/DH/Thuc Tap/pj/mlops', d)) and d.startswith('Model_v')]) + 1
        # model_version_name = f'Model_v{model_version}'

        # # Lưu kết quả
        # save_model_results(model_version_name, best_pipeline, metrics, summary_table, df_scorecard, final_threshold, csv_files)

if __name__ == '__main__':
    train_model()