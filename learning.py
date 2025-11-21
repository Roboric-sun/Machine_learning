import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import kagglehub

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ----------------------------- 0. ОПИСАНИЕ ЗАДАЧИ -----------------------------
# Регрессионная задача: по характеристикам дома/квартиры
# предсказать цену (столбец Price).


def main():
    # ----------------------------- 1. ЗАГРУЗКА ДАННЫХ -------------------------
    print("Скачиваю датасет...")
    dataset_path = kagglehub.dataset_download(
        "nguyentiennhan/vietnam-housing-dataset-2024"
    )
    print("Файлы скачаны в:", dataset_path)

    # ищем CSV-файл
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("В папке датасета нет .csv файлов")

    csv_path = os.path.join(dataset_path, csv_files[0])
    print("Использую файл:", csv_path)

    df = pd.read_csv(csv_path)
    print("Размер датасета:", df.shape)
    print(df.head())

    # целевая переменная
    TARGET_COL = "Price"

    # ----------------------- 2. EDA: ОБЩИЙ АНАЛИЗ ДАННЫХ ----------------------
    print("\nИнформация о данных:")
    print(df.info())

    print("\nСтатистика числовых признаков:")
    print(df.describe())

    print("\nРаспределение целевой переменной:")
    print(df[TARGET_COL].describe())

    # гистограммы числовых признаков
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

    if numeric_cols:
        df[numeric_cols + [TARGET_COL]].hist(figsize=(16, 10))
        plt.suptitle("Гистограммы числовых признаков")
        plt.tight_layout()
        plt.show()

        # корреляционная матрица
        corr = df[numeric_cols + [TARGET_COL]].corr()
        print("\nКорреляционная матрица:")
        print(corr)

        plt.figure(figsize=(10, 8))
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Корреляционная матрица")
        plt.tight_layout()
        plt.show()

    # ----------------------- 3. ОБРАБОТКА ПРОПУСКОВ ---------------------------
    print("\nПропуски по столбцам:")
    print(df.isna().sum())

    # отделяем адрес (текст, очень много уникальных значений – выкинем из модели)
    if "Address" in df.columns:
        df = df.drop(columns=["Address"])

    # заново определяем типы
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # числовые -> заполняем медианой
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # категориальные -> модой
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nПропуски после заполнения:")
    print(df.isna().sum())

    # ----------------------- 4. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ -----------------------
    print("\nКатегориальные признаки:", cat_cols)
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print("Размер после one-hot:", df_encoded.shape)

    # ----------------------- 5. НОРМАЛИЗАЦИЯ ПРИЗНАКОВ ------------------------
    # Для KNN нормализация важна, т.к. расстояние зависит от масштаба.
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]

    feature_cols = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # ----------------------- 6. TRAIN / TEST SPLIT ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nРазмер train:", X_train.shape, "Размер test:", X_test.shape)

    # ---------- 7–8. МОДЕЛЬ KNN + ПОДБОР ЧИСЛА СОСЕДЕЙ (РЕГРЕССИЯ) -----------
    k_values = range(1, 31)
    cv_scores = []

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="r2")
        cv_scores.append(scores.mean())

    best_k = k_values[int(np.argmax(cv_scores))]
    print("\nЛучшее k по CV:", best_k, "R2 =", max(cv_scores))

    knn_best = KNeighborsRegressor(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)

    y_train_pred = knn_best.predict(X_train)
    y_test_pred = knn_best.predict(X_test)

    def print_regression_metrics(name, y_true_train, y_pred_train, y_true_test, y_pred_test):
        print(f"\n====== {name} ======")
        for part, y_t, y_p in [
            ("TRAIN", y_true_train, y_pred_train),
            ("TEST", y_true_test, y_pred_test),
        ]:
            mse = mean_squared_error(y_t, y_p)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_t, y_p)
            r2 = r2_score(y_t, y_p)
            print(f"{part}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # 9. Ошибки на обучающей и тестовой выборках
    print_regression_metrics(
        f"KNN (k={best_k})",
        y_train, y_train_pred,
        y_test, y_test_pred
    )

    # --------------------- 10. ДРУГИЕ МОДЕЛИ (ЛИНЕЙНАЯ, RF) -------------------

    # 10.1 Линейная регрессия
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_train_lr = linreg.predict(X_train)
    y_test_lr = linreg.predict(X_test)
    print_regression_metrics("LinearRegression", y_train, y_train_lr, y_test, y_test_lr)

    # 10.2 Случайный лес
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_train_rf = rf.predict(X_train)
    y_test_rf = rf.predict(X_test)
    print_regression_metrics("RandomForestRegressor", y_train, y_train_rf, y_test, y_test_rf)

    # пункт 11 "несбалансированность классов" к регрессии не применим — классов нет.
    # Можно прямо так и написать в отчёте.

    # --------------------- 12. УДАЛЕНИЕ КОРРЕЛИРОВАННЫХ ПРИЗНАКОВ ------------
    # Смотрим только на признаки (без таргета)
    corr_matrix = pd.DataFrame(X_train, columns=feature_cols).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    print("\nСильно коррелирующие признаки (|corr| > 0.9), кандидаты на удаление:")
    print(to_drop)

    if to_drop:
        X_train_red = X_train.drop(columns=to_drop)
        X_test_red = X_test.drop(columns=to_drop)

        linreg_red = LinearRegression()
        linreg_red.fit(X_train_red, y_train)
        y_train_lr_red = linreg_red.predict(X_train_red)
        y_test_lr_red = linreg_red.predict(X_test_red)
        print_regression_metrics(
            "LinearRegression (после удаления коррелирующих признаков)",
            y_train, y_train_lr_red,
            y_test, y_test_lr_red
        )
    else:
        print("Признаков с сильной корреляцией не найдено.")

    # 13. Общие выводы пишешь в отчёте:
    # - какая модель показала лучший R2 / RMSE на тесте
    # - как повлияло k в KNN
    # - как сработало удаление коррелирующих признаков
    # - что видно из EDA (диапазоны цен, связи с площадью и т.п.)


if __name__ == "__main__":
    main()