{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNo9AiGGAHVLEe8Z/nNckd3",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ariz-tahoor/Titanic-dataset/blob/main/logistic%20model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Tqe1QqGcy_o0"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv(\"/content/titanic_preprocessed_new\")\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 504
        },
        "id": "4GVi0u8nzPgX",
        "outputId": "d1a42521-de0e-402d-e92e-36bf08610cfe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name       Age  SibSp  Parch  \\\n",
              "0                            Braund, Mr. Owen Harris -0.565736      1      0   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  0.663861      1      0   \n",
              "2                             Heikkinen, Miss. Laina -0.258337      0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  0.433312      1      0   \n",
              "4                           Allen, Mr. William Henry  0.433312      0      0   \n",
              "\n",
              "             Ticket      Fare  FamilySize  Sex_male  Title_Mr  Title_Mrs  \\\n",
              "0         A/5 21171 -0.312011    0.059160         1         1          0   \n",
              "1          PC 17599  2.461242    0.059160         0         0          1   \n",
              "2  STON/O2. 3101282 -0.282777   -0.560975         0         0          0   \n",
              "3            113803  1.673732    0.059160         0         0          1   \n",
              "4            373450 -0.277363   -0.560975         1         1          0   \n",
              "\n",
              "   Title_Other  Embarked_Q  Embarked_S  \n",
              "0            0           0           1  \n",
              "1            0           0           0  \n",
              "2            0           0           1  \n",
              "3            0           0           1  \n",
              "4            0           0           1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d536e6bf-2656-413f-b7ec-63afeb4a6bb4\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>FamilySize</th>\n",
              "      <th>Sex_male</th>\n",
              "      <th>Title_Mr</th>\n",
              "      <th>Title_Mrs</th>\n",
              "      <th>Title_Other</th>\n",
              "      <th>Embarked_Q</th>\n",
              "      <th>Embarked_S</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>-0.565736</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>-0.312011</td>\n",
              "      <td>0.059160</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>0.663861</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>2.461242</td>\n",
              "      <td>0.059160</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>-0.258337</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>-0.282777</td>\n",
              "      <td>-0.560975</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>0.433312</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>1.673732</td>\n",
              "      <td>0.059160</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>0.433312</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>-0.277363</td>\n",
              "      <td>-0.560975</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d536e6bf-2656-413f-b7ec-63afeb4a6bb4')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-d536e6bf-2656-413f-b7ec-63afeb4a6bb4 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-d536e6bf-2656-413f-b7ec-63afeb4a6bb4');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 891,\n  \"fields\": [\n    {\n      \"column\": \"PassengerId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 257,\n        \"min\": 1,\n        \"max\": 891,\n        \"num_unique_values\": 891,\n        \"samples\": [\n          710,\n          440,\n          841\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Survived\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Pclass\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 1,\n        \"max\": 3,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          3,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Name\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 891,\n        \"samples\": [\n          \"Moubarek, Master. Halim Gonios (\\\"William George\\\")\",\n          \"Kvillner, Mr. Johan Henrik Johannesson\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.000561640033049,\n        \"min\": -2.224156079948338,\n        \"max\": 3.8915544515600136,\n        \"num_unique_values\": 88,\n        \"samples\": [\n          -2.1987956316523123,\n          -0.5657364610748746\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"SibSp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 8,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Parch\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Ticket\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 681,\n        \"samples\": [\n          \"11774\",\n          \"248740\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Fare\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.1521996308806055,\n        \"min\": -0.6260047813734322,\n        \"max\": 21.56273820248077,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          -0.1391319035409881,\n          1.6201363384380845\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"FamilySize\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.000561640033051,\n        \"min\": -0.5609748304808925,\n        \"max\": 5.64037224096421,\n        \"num_unique_values\": 9,\n        \"samples\": [\n          3.779968119530679,\n          -0.5609748304808925\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex_male\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Title_Mr\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Title_Mrs\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Title_Other\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked_Q\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Embarked_S\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 60
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x = df.drop(columns = [\"PassengerId\", \"Survived\", \"Name\", \"Ticket\"])\n",
        "y = df[\"Survived\"]\n",
        "from sklearn.model_selection import train_test_split\n",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=67, stratify = y)\n",
        "\n",
        "print(x_train.shape, y_train.shape)\n",
        "print(x_test.shape, y_test.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aDZWTm2RzVDS",
        "outputId": "5aa811f9-2b91-4535-bb14-60d5a1d67c11"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(712, 12) (712,)\n",
            "(179, 12) (179,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "model = LogisticRegression()\n",
        "\n",
        "model.fit(x_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 80
        },
        "id": "ShfjRy7Mz84P",
        "outputId": "cd1893e0-81ea-4a05-f14d-15f32239b324"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LogisticRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-4 {\n",
              "  /* Definition of color scheme common for light and dark mode */\n",
              "  --sklearn-color-text: #000;\n",
              "  --sklearn-color-text-muted: #666;\n",
              "  --sklearn-color-line: gray;\n",
              "  /* Definition of color scheme for unfitted estimators */\n",
              "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
              "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
              "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
              "  --sklearn-color-unfitted-level-3: chocolate;\n",
              "  /* Definition of color scheme for fitted estimators */\n",
              "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
              "  --sklearn-color-fitted-level-1: #d4ebff;\n",
              "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
              "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
              "\n",
              "  /* Specific color for light theme */\n",
              "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
              "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-icon: #696969;\n",
              "\n",
              "  @media (prefers-color-scheme: dark) {\n",
              "    /* Redefinition of color scheme for dark theme */\n",
              "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
              "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-icon: #878787;\n",
              "  }\n",
              "}\n",
              "\n",
              "#sk-container-id-4 {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 pre {\n",
              "  padding: 0;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 input.sk-hidden--visually {\n",
              "  border: 0;\n",
              "  clip: rect(1px 1px 1px 1px);\n",
              "  clip: rect(1px, 1px, 1px, 1px);\n",
              "  height: 1px;\n",
              "  margin: -1px;\n",
              "  overflow: hidden;\n",
              "  padding: 0;\n",
              "  position: absolute;\n",
              "  width: 1px;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-dashed-wrapped {\n",
              "  border: 1px dashed var(--sklearn-color-line);\n",
              "  margin: 0 0.4em 0.5em 0.4em;\n",
              "  box-sizing: border-box;\n",
              "  padding-bottom: 0.4em;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-container {\n",
              "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
              "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
              "     so we also need the `!important` here to be able to override the\n",
              "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
              "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
              "  display: inline-block !important;\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-text-repr-fallback {\n",
              "  display: none;\n",
              "}\n",
              "\n",
              "div.sk-parallel-item,\n",
              "div.sk-serial,\n",
              "div.sk-item {\n",
              "  /* draw centered vertical line to link estimators */\n",
              "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
              "  background-size: 2px 100%;\n",
              "  background-repeat: no-repeat;\n",
              "  background-position: center center;\n",
              "}\n",
              "\n",
              "/* Parallel-specific style estimator block */\n",
              "\n",
              "#sk-container-id-4 div.sk-parallel-item::after {\n",
              "  content: \"\";\n",
              "  width: 100%;\n",
              "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
              "  flex-grow: 1;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-parallel {\n",
              "  display: flex;\n",
              "  align-items: stretch;\n",
              "  justify-content: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-parallel-item {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-parallel-item:first-child::after {\n",
              "  align-self: flex-end;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-parallel-item:last-child::after {\n",
              "  align-self: flex-start;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-parallel-item:only-child::after {\n",
              "  width: 0;\n",
              "}\n",
              "\n",
              "/* Serial-specific style estimator block */\n",
              "\n",
              "#sk-container-id-4 div.sk-serial {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "  align-items: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  padding-right: 1em;\n",
              "  padding-left: 1em;\n",
              "}\n",
              "\n",
              "\n",
              "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
              "clickable and can be expanded/collapsed.\n",
              "- Pipeline and ColumnTransformer use this feature and define the default style\n",
              "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
              "*/\n",
              "\n",
              "/* Pipeline and ColumnTransformer style (default) */\n",
              "\n",
              "#sk-container-id-4 div.sk-toggleable {\n",
              "  /* Default theme specific background. It is overwritten whether we have a\n",
              "  specific estimator or a Pipeline/ColumnTransformer */\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "/* Toggleable label */\n",
              "#sk-container-id-4 label.sk-toggleable__label {\n",
              "  cursor: pointer;\n",
              "  display: flex;\n",
              "  width: 100%;\n",
              "  margin-bottom: 0;\n",
              "  padding: 0.5em;\n",
              "  box-sizing: border-box;\n",
              "  text-align: center;\n",
              "  align-items: start;\n",
              "  justify-content: space-between;\n",
              "  gap: 0.5em;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 label.sk-toggleable__label .caption {\n",
              "  font-size: 0.6rem;\n",
              "  font-weight: lighter;\n",
              "  color: var(--sklearn-color-text-muted);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 label.sk-toggleable__label-arrow:before {\n",
              "  /* Arrow on the left of the label */\n",
              "  content: \"▸\";\n",
              "  float: left;\n",
              "  margin-right: 0.25em;\n",
              "  color: var(--sklearn-color-icon);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "/* Toggleable content - dropdown */\n",
              "\n",
              "#sk-container-id-4 div.sk-toggleable__content {\n",
              "  max-height: 0;\n",
              "  max-width: 0;\n",
              "  overflow: hidden;\n",
              "  text-align: left;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-toggleable__content.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-toggleable__content pre {\n",
              "  margin: 0.2em;\n",
              "  border-radius: 0.25em;\n",
              "  color: var(--sklearn-color-text);\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-toggleable__content.fitted pre {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
              "  /* Expand drop-down */\n",
              "  max-height: 200px;\n",
              "  max-width: 100%;\n",
              "  overflow: auto;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
              "  content: \"▾\";\n",
              "}\n",
              "\n",
              "/* Pipeline/ColumnTransformer-specific style */\n",
              "\n",
              "#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator-specific style */\n",
              "\n",
              "/* Colorize estimator box */\n",
              "#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-label label.sk-toggleable__label,\n",
              "#sk-container-id-4 div.sk-label label {\n",
              "  /* The background is the default theme color */\n",
              "  color: var(--sklearn-color-text-on-default-background);\n",
              "}\n",
              "\n",
              "/* On hover, darken the color of the background */\n",
              "#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "/* Label box, darken color on hover, fitted */\n",
              "#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator label */\n",
              "\n",
              "#sk-container-id-4 div.sk-label label {\n",
              "  font-family: monospace;\n",
              "  font-weight: bold;\n",
              "  display: inline-block;\n",
              "  line-height: 1.2em;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-label-container {\n",
              "  text-align: center;\n",
              "}\n",
              "\n",
              "/* Estimator-specific */\n",
              "#sk-container-id-4 div.sk-estimator {\n",
              "  font-family: monospace;\n",
              "  border: 1px dotted var(--sklearn-color-border-box);\n",
              "  border-radius: 0.25em;\n",
              "  box-sizing: border-box;\n",
              "  margin-bottom: 0.5em;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-estimator.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "/* on hover */\n",
              "#sk-container-id-4 div.sk-estimator:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-4 div.sk-estimator.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
              "\n",
              "/* Common style for \"i\" and \"?\" */\n",
              "\n",
              ".sk-estimator-doc-link,\n",
              "a:link.sk-estimator-doc-link,\n",
              "a:visited.sk-estimator-doc-link {\n",
              "  float: right;\n",
              "  font-size: smaller;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1em;\n",
              "  height: 1em;\n",
              "  width: 1em;\n",
              "  text-decoration: none !important;\n",
              "  margin-left: 0.5em;\n",
              "  text-align: center;\n",
              "  /* unfitted */\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted,\n",
              "a:link.sk-estimator-doc-link.fitted,\n",
              "a:visited.sk-estimator-doc-link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "/* Span, style for the box shown on hovering the info icon */\n",
              ".sk-estimator-doc-link span {\n",
              "  display: none;\n",
              "  z-index: 9999;\n",
              "  position: relative;\n",
              "  font-weight: normal;\n",
              "  right: .2ex;\n",
              "  padding: .5ex;\n",
              "  margin: .5ex;\n",
              "  width: min-content;\n",
              "  min-width: 20ex;\n",
              "  max-width: 50ex;\n",
              "  color: var(--sklearn-color-text);\n",
              "  box-shadow: 2pt 2pt 4pt #999;\n",
              "  /* unfitted */\n",
              "  background: var(--sklearn-color-unfitted-level-0);\n",
              "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted span {\n",
              "  /* fitted */\n",
              "  background: var(--sklearn-color-fitted-level-0);\n",
              "  border: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link:hover span {\n",
              "  display: block;\n",
              "}\n",
              "\n",
              "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
              "\n",
              "#sk-container-id-4 a.estimator_doc_link {\n",
              "  float: right;\n",
              "  font-size: 1rem;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1rem;\n",
              "  height: 1rem;\n",
              "  width: 1rem;\n",
              "  text-decoration: none;\n",
              "  /* unfitted */\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 a.estimator_doc_link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "#sk-container-id-4 a.estimator_doc_link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "#sk-container-id-4 a.estimator_doc_link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>LogisticRegression</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html\">?<span>Documentation for LogisticRegression</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></div></label><div class=\"sk-toggleable__content fitted\"><pre>LogisticRegression()</pre></div> </div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = model.predict(x_test)\n",
        "print(y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MiBU-FYz0KYB",
        "outputId": "d612ae05-007a-44e1-d519-14671fea5a77"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0 1 0 0 1 1 0 0 1 0 1 0 0 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0\n",
            " 1 1 1 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 1 0 0 0 0 1 1 1 0 0\n",
            " 0 0 1 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 0\n",
            " 1 0 0 1 0 1 1 0 1 1 1 0 0 1 0 0 1 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0\n",
            " 0 1 1 0 1 0 1 1 0 0 0 1 1 0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 1 1 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "print(\"accuracy: \", accuracy_score(y_test, y_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pdf3fqbN0Rmp",
        "outputId": "cc8533d2-c9c0-4bf7-995e-6f92091334f9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "accuracy:  0.7932960893854749\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import precision_score\n",
        "print(\"precision: \", precision_score(y_test, y_pred)) #gives false positives"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4Cqz7bwF2baY",
        "outputId": "5b5420d1-abd2-4f59-8cac-b5e7222180a2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "precision:  0.7352941176470589\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import recall_score\n",
        "print(\"recall: \", recall_score(y_test, y_pred)) #gives false negatives"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NHvPnd8z2wmP",
        "outputId": "49b2ee26-36c9-4edb-9194-407557ac91c8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "recall:  0.7246376811594203\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import f1_score\n",
        "print(\"f1 score: \", f1_score(y_test, y_pred)) #harmonic mean of recall and precision"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4VVDDml53mtj",
        "outputId": "3a5544f8-4978-49ee-d293-fa07d9cdae63"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "f1 score:  0.7299270072992701\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import classification_report\n",
        "print(classification_report(y_test, y_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sW3rOImY5hJc",
        "outputId": "288d4b9b-c7a9-42a3-951c-93a0a2b60214"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.83      0.84      0.83       110\n",
            "           1       0.74      0.72      0.73        69\n",
            "\n",
            "    accuracy                           0.79       179\n",
            "   macro avg       0.78      0.78      0.78       179\n",
            "weighted avg       0.79      0.79      0.79       179\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "print(cm)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EWtZ2dS65l0y",
        "outputId": "3fb47acd-b8e5-4dee-f436-217133e4cb5a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[92 18]\n",
            " [19 50]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[\"Died\", \"Survived\"])\n",
        "disp.plot(cmap=\"Blues\")\n",
        "plt.title(\"Confusion Matrix - Logistic Regression\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "ukcpkVNt6DbM",
        "outputId": "d7d5c6d7-5968-4e5b-b0d6-ccc10bb20ef1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiYAAAHHCAYAAACLPpP8AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAVntJREFUeJzt3XdcU1f/B/DPDSMgkCAOAjJEUMFdsCrugeIsFqyLtjjbqrWuulpxK4q1WltXq4JaaV2Valu17om2btviwIkyrAMCKPv+/vBHnkZAExIgwc+7r/t6ntx77jnnxgS+fM859wqiKIogIiIiMgCS8u4AERERUQEGJkRERGQwGJgQERGRwWBgQkRERAaDgQkREREZDAYmREREZDAYmBAREZHBYGBCREREBoOBCRERERkMBiZU7q5fv44uXbpALpdDEARER0frtf7bt29DEARERkbqtV5j1r59e7Rv3768u1FmDh8+DEEQcPjwYb3UFxkZCUEQcPv2bb3UR8DMmTMhCEJ5d4MMAAMTAgDcuHEDH374IWrVqgULCwvIZDK0atUKX331FZ49e1aqbYeEhODy5cuYN28eNm7ciKZNm5Zqe2Vp0KBBEAQBMpmsyPfx+vXrEAQBgiDgiy++0Lr+hIQEzJw5ExcuXNBDb8tGzZo10bNnz/Luhkbmz5+v90D5RQVBTsFmamqKGjVqYNCgQbh//36ptk1kiEzLuwNU/n799Ve88847kEqleP/999GgQQNkZ2fj+PHjmDhxIv7++298++23pdL2s2fPEBMTg88//xwff/xxqbTh6uqKZ8+ewczMrFTqfxVTU1M8ffoUu3btQt++fdWObdq0CRYWFsjMzCxR3QkJCZg1axZq1qyJJk2aaHze77//XqL2jFXbtm3x7NkzmJuba3Xe/Pnz0adPH/Tu3Vtt/3vvvYf+/ftDKpXqrY+zZ8+Gm5sbMjMzcerUKURGRuL48eP466+/YGFhobd2DNW0adMwZcqU8u4GGQAGJq+5W7duoX///nB1dcXBgwfh4OCgOjZq1CjExcXh119/LbX2//33XwCAra1tqbUhCEK5/mCXSqVo1aoVfvjhh0KBSVRUFHr06IHt27eXSV+ePn2KSpUqaf0L2thJJBK9fgZMTExgYmKit/oAoFu3bqps4bBhw1C1alUsXLgQO3fuLPS5KU2iKCIzMxOWlpZl1ibwPIA3NeWvJOJQzmsvPDwc6enpWLt2rVpQUsDDwwNjxoxRvc7NzcWcOXPg7u4OqVSKmjVr4rPPPkNWVpbaeQXp+uPHj6NZs2awsLBArVq1sGHDBlWZmTNnwtXVFQAwceJECIKAmjVrAng+BFLw//+rqHHoffv2oXXr1rC1tYW1tTXq1q2Lzz77THW8uDkmBw8eRJs2bWBlZQVbW1sEBAQgNja2yPbi4uIwaNAg2NraQi6XY/DgwXj69Gnxb+wLBg4ciN27dyMlJUW1788//8T169cxcODAQuUfP36MTz/9FA0bNoS1tTVkMhm6deuGixcvqsocPnwYb775JgBg8ODBqqGAguts3749GjRogLNnz6Jt27aoVKmS6n15cY5JSEgILCwsCl2/v78/KleujISEBI2vVR80/Zzl5+dj5syZcHR0RKVKldChQwf8888/qFmzJgYNGqQqV9Qck+vXryMoKAgKhQIWFhZwcnJC//79kZqaCuB5QJuRkYH169er3tuCOoubY7J79260a9cONjY2kMlkePPNNxEVFVWi96BNmzYAng+z/teVK1fQp08f2NnZwcLCAk2bNsXOnTsLnX/p0iW0a9cOlpaWcHJywty5cxEREVGo3wXf1b1796Jp06awtLTE6tWrAQApKSkYO3YsnJ2dIZVK4eHhgYULFyI/P1+trR9//BE+Pj6q627YsCG++uor1fGcnBzMmjULtWvXhoWFBapUqYLWrVtj3759qjJFfbf1+fOGjAfD09fcrl27UKtWLbRs2VKj8sOGDcP69evRp08fTJgwAadPn0ZYWBhiY2OxY8cOtbJxcXHo06cPhg4dipCQEKxbtw6DBg2Cj48P6tevj8DAQNja2mLcuHEYMGAAunfvDmtra636//fff6Nnz55o1KgRZs+eDalUiri4OJw4ceKl5+3fvx/dunVDrVq1MHPmTDx79gxff/01WrVqhXPnzhUKivr27Qs3NzeEhYXh3LlzWLNmDapXr46FCxdq1M/AwEB89NFH+OmnnzBkyBAAz7Mlnp6e8Pb2LlT+5s2biI6OxjvvvAM3NzckJydj9erVaNeuHf755x84OjrCy8sLs2fPxvTp0/HBBx+ofpH999/y0aNH6NatG/r37493330X9vb2Rfbvq6++wsGDBxESEoKYmBiYmJhg9erV+P3337Fx40Y4OjpqdJ36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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_proba = model.predict_proba(x_test)[:,1]"
      ],
      "metadata": {
        "id": "HQLYuoOE7f35"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay\n",
        "fpr, tpr, thresholds = roc_curve(y_test, y_proba)\n",
        "roc_auc = roc_auc_score(y_test, y_proba)\n",
        "print(\"auc: \", roc_auc)\n",
        "print(\"fpr: \", fpr)\n",
        "print(\"tpr: \", tpr)\n",
        "print(\"thresholds: \", thresholds)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Kex36tSz7vs-",
        "outputId": "1be40e12-5394-44a9-d954-5a6f8b7f630b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "auc:  0.8466403162055336\n",
            "fpr:  [0.         0.         0.         0.00909091 0.00909091 0.02727273\n",
            " 0.02727273 0.03636364 0.03636364 0.04545455 0.04545455 0.06363636\n",
            " 0.06363636 0.07272727 0.07272727 0.08181818 0.08181818 0.1\n",
            " 0.1        0.11818182 0.11818182 0.12727273 0.12727273 0.15454545\n",
            " 0.15454545 0.16363636 0.16363636 0.18181818 0.18181818 0.20909091\n",
            " 0.20909091 0.28181818 0.29090909 0.3        0.3        0.43636364\n",
            " 0.43636364 0.46363636 0.46363636 0.53636364 0.53636364 0.55454545\n",
            " 0.56363636 0.58181818 0.6        0.66363636 0.66363636 0.68181818\n",
            " 0.7        0.7        0.75454545 0.75454545 0.76363636 0.76363636\n",
            " 0.79090909 0.80909091 0.84545455 0.84545455 1.        ]\n",
            "tpr:  [0.         0.01449275 0.26086957 0.26086957 0.46376812 0.46376812\n",
            " 0.47826087 0.47826087 0.49275362 0.49275362 0.52173913 0.52173913\n",
            " 0.53623188 0.53623188 0.5942029  0.5942029  0.62318841 0.62318841\n",
            " 0.63768116 0.63768116 0.65217391 0.65217391 0.66666667 0.66666667\n",
            " 0.72463768 0.72463768 0.73913043 0.73913043 0.75362319 0.75362319\n",
            " 0.7826087  0.7826087  0.79710145 0.79710145 0.85507246 0.85507246\n",
            " 0.86956522 0.86956522 0.88405797 0.88405797 0.89855072 0.89855072\n",
            " 0.91304348 0.91304348 0.91304348 0.91304348 0.92753623 0.92753623\n",
            " 0.92753623 0.94202899 0.94202899 0.95652174 0.95652174 0.97101449\n",
            " 0.97101449 0.97101449 0.97101449 1.         1.        ]\n",
            "thresholds:  [       inf 0.96678273 0.88979287 0.88040837 0.78519676 0.78390571\n",
            " 0.7820769  0.76941488 0.75584387 0.75328079 0.73547052 0.72907156\n",
            " 0.72828141 0.70360056 0.67649226 0.66913702 0.6569971  0.65266356\n",
            " 0.65197969 0.64574249 0.6199291  0.61913669 0.61540164 0.56373649\n",
            " 0.50743216 0.50179621 0.48936404 0.42039171 0.41121017 0.36307521\n",
            " 0.30471713 0.2665325  0.26250339 0.26028454 0.24474033 0.16990731\n",
            " 0.16794945 0.15222056 0.14997226 0.12334357 0.12140366 0.11781437\n",
            " 0.11761525 0.11758945 0.11688596 0.1138385  0.11322036 0.11156234\n",
            " 0.11034525 0.11032269 0.09810598 0.09766443 0.09426263 0.09149975\n",
            " 0.08792372 0.08791254 0.08096804 0.08021136 0.0024945 ]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "RocCurveDisplay.from_predictions(y_test, y_proba, name=\"Logistic Regression\")\n",
        "plt.plot([0, 1], [0, 1], linestyle=\"--\", color=\"gray\", label=\"Random guess (AUC=0.5)\")\n",
        "plt.title(\"ROC Curve - Logistic Regression\")\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "whmZjqBz73Lc",
        "outputId": "e7a25ca2-6aaa-4312-c4ef-84619f560562"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAcAAAAHHCAYAAAAoIIjLAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAhV9JREFUeJzt3XdcU9f7B/BPAiTsJSBDFJyAC0XBbVUU96qK4gBr3atSq1gHauuqC1tXnWir4qha3EUrFRG3OBEVERygIhJkQ3J+f/Djfg0ETCAhjOf9euWlOffce5/chDw5955zD48xxkAIIYRUM3x1B0AIIYSoAyVAQggh1RIlQEIIIdUSJUBCCCHVEiVAQggh1RIlQEIIIdUSJUBCCCHVEiVAQggh1RIlQEIIIdUSJUBCqgkej4fFixcrZVsvXrwAj8dDYGCgUrZHgNDQUPB4PISGhqo7lGqDEmAVERgYCB6Pxz00NTVhY2MDHx8fvH79WuY6jDH88ccf6NSpE4yNjaGrq4umTZti6dKlSE9PL3Zfx44dQ69evWBmZgaBQABra2sMGzYM//77r1yxZmVlYf369XBzc4ORkRG0tbXRsGFDTJs2DU+ePCnV669MfHx8oK+vr+4w5LJ//34EBASodB8FybTgwefzYWpqil69eiEiIkKl+ybVG4/uBVo1BAYGYuzYsVi6dCns7e2RlZWFq1evIjAwEHZ2dnjw4AG0tbW5+mKxGF5eXjh06BA6duyIwYMHQ1dXF2FhYdi/fz+cnJxw/vx51KxZk1uHMYZvvvkGgYGBaNGiBYYMGQJLS0skJCTg2LFjuHXrFsLDw9GuXbti40xKSkLPnj1x69Yt9O3bF+7u7tDX10d0dDSCgoKQmJiInJwclR4rdfPx8cGRI0eQlpZWrvvNysqCpqYmNDU15V6nb9++ePDgAV68eCFVzhhDdnY2tLS0oKGhUaa4Xrx4AXt7e4wYMQK9e/eGWCzGkydPsHnzZmRmZuLGjRto2rRpmfZRGUgkEuTk5EAgEIDPp7ZJuWCkSti9ezcDwG7cuCFVPnfuXAaAHTx4UKp8+fLlDACbPXt2kW0FBwczPp/PevbsKVW+evVqBoB99913TCKRFFlv79697Nq1ayXG2adPH8bn89mRI0eKLMvKymLff/99ievLKzc3l2VnZytlW8rm7e3N9PT01B2GXPr06cPq1Kmj0n3ExsYyAGz16tVS5WfOnGEA2OTJk1W6f1nS0tLKfZ+k/FECrCKKS4AnT55kANjy5cu5soyMDGZiYsIaNmzIcnNzZW5v7NixDACLiIjg1jE1NWUODg4sLy+vVDFevXqVAWDjx4+Xq37nzp1Z586di5R7e3tLfSl//gW6fv16VrduXcbn89nVq1eZhoYGW7x4cZFtPH78mAFgv/32G1f28eNHNnPmTFarVi0mEAhYvXr12MqVK5lYLFb4tZZE3gR46NAh1rJlS6atrc1q1KjBRo4cyV69eiWznqOjIxMKhaxx48bs6NGjRY4RY4wBYP7+/tzz1NRUNnPmTFanTh0mEAiYubk5c3d3Z7du3WKM5R9/AFKPgm0WHPPdu3dL7SMqKooNHTqUmZmZMW1tbdawYUP2448/lvg6i0uAaWlpDADr0aOHVLm871NSUhIbNWoUMzAwYEZGRmzMmDEsMjKySNwF78ezZ89Yr169mL6+PhswYABjjDGxWMzWr1/PnJycmFAoZBYWFmzChAksOTlZal83btxgPXr0YDVq1GDa2trMzs6OjR07VqrOgQMHWMuWLZm+vj4zMDBgTZo0YQEBAdzyixcvMgDs4sWLUuvJ8zkoeA2vXr1iAwYMYHp6eszMzIx9//33pf57rQ7kPxdCKqWCU1cmJiZc2eXLl/Hx40fMnDmz2NNhY8aMwe7du3Hy5Em0adMGly9fRnJyMr777rtSn/IKDg4GAIwePbpU63/J7t27kZWVhQkTJkAoFMLKygqdO3fGoUOH4O/vL1X34MGD0NDQwNChQwEAGRkZ6Ny5M16/fo2JEyeidu3auHLlCubNm4eEhASVXwcrrOCUduvWrbFixQq8ffsWGzZsQHh4OO7cuQNjY2MAwKlTp+Dp6YmmTZtixYoV+PjxI8aNGwcbG5sv7mPSpEk4cuQIpk2bBicnJ3z48AGXL19GVFQUWrZsifnz50MkEuHVq1dYv349AJR47fLevXvo2LEjtLS0MGHCBNjZ2SEmJgYnTpzAsmXLFD4Gsj678r5PEokE/fr1w/Xr1zF58mQ4ODjg77//hre3t8x95eXlwcPDAx06dMCaNWugq6sLAJg4cSL3XsyYMQOxsbHYuHEj7ty5g/DwcGhpaeHdu3fo0aMHzM3N4efnB2NjY7x48QJHjx7lth8SEoIRI0agW7duWLVqFQAgKioK4eHhmDlzZrHHQN7PAZB/WcPDwwNubm5Ys2YNzp8/j7Vr16JevXqYPHmywse/WlB3BibKUdACPH/+PHv//j17+fIlO3LkCDM3N2dCoZC9fPmSqxsQEMAAsGPHjhW7veTkZAaADR48mDHG2IYNG764zpcMGjSIAWAfP36Uq76iLUBDQ0P27t07qbq///47A8Du378vVe7k5MS6du3KPf/pp5+Ynp4ee/LkiVQ9Pz8/pqGhweLj4+WKWR5fagHm5OQwCwsL1qRJE5aZmcmVF7TmFy1axJU1bdqU1apVi3369IkrCw0NlWqtFUChFqCRkRGbOnVqibEWdwpUVguwU6dOzMDAgMXFxUnVlXW6XNa2lixZwt6/f88SExNZWFgYa926NQPADh8+zNWV933666+/GACpFpZYLGZdu3aV2QIEwPz8/KS2GRYWxgCwffv2SZWfPXtWqvzYsWMyz758bubMmczQ0LDE1ljhFqAin4OC17B06VKpbbZo0YK5uLgUu8/qjq60VjHu7u4wNzeHra0thgwZAj09PQQHB6NWrVpcnU+fPgEADAwMit1OwbLU1FSpf0ta50uUsY2SfP311zA3N5cqGzx4MDQ1NXHw4EGu7MGDB3j06BE8PT25ssOHD6Njx44wMTFBUlIS93B3d4dYLMalS5dUErMsN2/exLt37zBlyhSpjkt9+vSBg4MDTp06BQB48+YN7t+/jzFjxki1zDp37ixXpxFjY2Ncu3YNb968KXPM79+/x6VLl/DNN9+gdu3aUst4PJ5c2/D394e5uTksLS3RsWNHREVFYe3atRgyZAhXR9736ezZs9DS0sL48eO5dfl8PqZOnVrs/gu3kg4fPgwjIyN0795dal8uLi7Q19fHxYsXAYBrhZ08eRK5ubkyt21sbIz09HSEhITIdSwA+T8Hn5s0aZLU844dO+L58+dy77O6oQRYxWzatAkhISE4cuQIevfujaSkJAiFQqk6BQmoIBHKUjhJGhoafnGdL1HGNkpib29fpMzMzAzdunXDoUOHuLKDBw9CU1MTgwcP5sqePn2Ks2fPwtzcXOrh7u4OAHj37l2x+xWJREhMTOQeycnJZXodcXFxAIBGjRoVWebg4MAtL/i3fv36RerJKivsl19+wYMHD2BrawtXV1csXry41F+WBes1adKkVOsDwIQJExASEoITJ05g1qxZyMzMhFgslqoj7/sUFxcHKysr7lRmgeKOi6amptSPxIJ9iUQiWFhYFNlfWloat6/OnTvj66+/xpIlS2BmZoYBAwZg9+7dyM7O5rY1ZcoUNGzYEL169UKtWrXwzTff4OzZsyUeD3k/BwW0tbWL/AA0MTHBx48fS9xPdUbXAKsYV1dXtGrVCgAwcOBAdOjQAV5eXoiOjuZaCY6OjgDyr9kMHDhQ5nbu3bsHAHBycgKQ/wcHAPfv3y92nS/5fBsdO3b8Yn0ejwcmY5RO4S/FAjo6OjLLhw8fjrFjxyIyMhLOzs44dOgQunXrBjMzM66ORCJB9+7dMWfOHJnbaNiwYbFxzpw5E3v27OGed+7cuVIMZh42bBg6duyIY8eO4Z9//sHq1auxatUqHD16FL169Sr3eBo0aMAlsr59+0JDQwN+fn7o0qUL95kuy/tUEqFQWGTogUQigYWFBfbt2ydznYJkw+PxcOTIEVy9ehUnTpzAuXPn8M0332Dt2rW4evUq9PX1YWFhgcjISJw7dw5nzpzBmTNnsHv3bowZM0bqs1MWZR2OUi2p+xwsUY7ieoEWXFdYsWIFV5aens6MjY1Zo0aNir0m8c0330j1Ak1PT2cmJibM0dGx1L3Krly5wgCwCRMmyFV/0KBBrHnz5kXKO3bsWGwvUFk+fvzIBAIB8/PzY3fu3JHZe9HJyYm1bdtW3pci5eHDhywkJIR73Lx5s8T6X7oGWHCcNm/eXGSZo6Mjd03n9evXDIDMXpZNmzb94jXAwt6+fctsbGxY+/btubK+ffvKdQ3w3bt3DACbOXNmsdsvTnHv38ePH5mRkRHz8PDgyuR9n8aPH8+0tLRYenq6VHnBtUFZvUALmzJlCtPQ0GAZGRkKviLG9u3bxwCw7du3y1wuFovZxIkTGQD29OlTxljRa4Dyfg5Keg3+/v6MvuaLR6dAq7ivvvoKrq6uCAgIQFZWFgBAV1cXs2fPRnR0NObPn19knVOnTiEwMBAeHh5o06YNt87cuXMRFRWFuXPnymyZ/fnnn7h+/XqxsbRt2xY9e/bEjh07cPz48SLLc3JyMHv2bO55vXr18PjxY7x//54ru3v3LsLDw+V+/UD+9RcPDw8cOnQIQUFBEAgERVqxw4YNQ0REBM6dO1dk/ZSUFOTl5RW7fScnJ7i7u3MPFxcXheIrrFWrVrCwsMDWrVulTqOdOXMGUVFR6NOnDwDA2toaTZo0wd69e6UG1f/333+4f/9+ifsQi8UQiURSZRYWFrC2tpbap56eXpF6spibm6NTp07YtWsX4uPjpZbJ+qzIw9jYGBMnTsS5c+cQGRkJQP73ycPDA7m5udi+fTu3XCKRYNOmTXLvf9iwYRCLxfjpp5+KLMvLy0NKSgoA4OPHj0Veo7OzMwBwx/LDhw9Sy/l8Ppo1ayZVpzB5Pwek9OgUaDXwww8/YOjQoQgMDOQukvv5+eHOnTtYtWoVIiIi8PXXX0NHRweXL1/Gn3/+CUdHxyKnZn744Qc8fPgQa9euxcWLF7k7wSQmJuL48eO4fv06rly5UmIse/fuRY8ePTB48GD069cP3bp1g56eHp4+fYqgoCAkJCRgzZo1AIBvvvkG69atg4eHB8aNG4d3795h69ataNy4MdehRl6enp4YNWoUNm/eDA8PD6nu4wWvLTg4GH379oWPjw9cXFyQnp6O+/fv48iRI3jx4oXUKdOyys3Nxc8//1yk3NTUFFOmTMGqVaswduxYdO7cGSNGjOC6v9vZ2WHWrFlc/eXLl2PAgAFo3749xo4di48fP2Ljxo1o0qRJiXea+fTpE2rVqoUhQ4agefPm0NfXx/nz53Hjxg2sXbuWq+fi4oKDBw/C19cXrVu3hr6+Pvr16ydzm7/++is6dOiAli1bYsKECbC3t8eLFy9w6tQpLoEpaubMmQgICMDKlSsRFBQk9/s0cOBAuLq64vvvv8ezZ8/g4OCA4OBg7vqsPB1zOnfujIkTJ2LFihWIjIxEjx49oKWlhadPn+Lw4cPYsGEDhgwZgj179mDz5s0YNGgQ6tWrh0+fPmH79u0wNDRE7969AQDffvstkpOT0bVrV9SqVQtxcXH47bff4OzszF2SKExLS0vuzwEpJTW3QImSFHcKlLH80y316tVj9erVkzp9KRaL2e7du1n79u2ZoaEh09bWZo0bN2ZLliwp8U4YR44cYT169GCmpqZMU1OTWVlZMU9PTxYaGipXrBkZGWzNmjWsdevWTF9fnwkEAtagQQM2ffp09uzZM6m6f/75J6tbty4TCATM2dmZnTt3rsSB8MVJTU1lOjo6DAD7888/Zdb59OkTmzdvHqtfvz4TCATMzMyMtWvXjq1Zs4bl5OTI9drkUdBlXdajXr16XL2DBw+yFi1aMKFQyExNTYsdCB8UFMQcHByYUChkTZo0YcHBwezrr79mDg4OUvXw2SnQ7Oxs9sMPP7DmzZszAwMDpqenx5o3b17kdFtaWhrz8vJixsbGcg2Ef/DgARs0aBAzNjZm2trarFGjRmzhwoUlHo8vvX8+Pj5MQ0OD+2zI+z69f/+eeXl5cQPhfXx8WHh4OAPAgoKCpN6Pkk5Jb9u2jbm4uDAdHR1mYGDAmjZtyubMmcPevHnDGGPs9u3bbMSIEax27drcYPm+fftKnQov+JuxsLBgAoGA1a5dm02cOJElJCRwdYobCC/P54BOgZYO3QuUkCrI2dkZ5ubmCnW7rw6OHz+OQYMG4fLly2jfvr26wyFqRtcACanEcnNzi1yfDA0Nxd27d/HVV1+pJ6gKIjMzU+q5WCzGb7/9BkNDQ7Rs2VJNUZGKhK4BElKJvX79Gu7u7hg1ahSsra3x+PFjbN26FZaWlkUGRVc306dPR2ZmJtq2bYvs7GwcPXoUV65cwfLly4sdMkOqFzoFSkglJhKJMGHCBISHh+P9+/fQ09NDt27dsHLlStSrV0/d4anV/v37sXbtWjx79gxZWVmoX78+Jk+ejGnTpqk7NFJBUAIkhBBSLdE1QEIIIdUSJUBCCCHVklo7wVy6dAmrV6/GrVu3kJCQgGPHjn3xPpOhoaHw9fXFw4cPYWtriwULFsDHx0fufUokErx58wYGBgZy36WeEEJIxcEYw6dPn2BtbV3kHq6KUGsCTE9PR/PmzfHNN99I3Zm/OLGxsejTpw8mTZqEffv24cKFC/j2229hZWUFDw8Pufb55s0b2NraljV0Qgghavby5csis3goosJ0guHxeF9sAc6dOxenTp3CgwcPuLLhw4cjJSXli1OLFBCJRDA2NsbLly+56XkIIYRUHqmpqbC1tUVKSgqMjIxKvZ1KNQ4wIiKCmy6lgIeHB7777ju5t1Fw2tPQ0JASICGkSkjPzsP1F8mQSCpEe0ZlmtUyhrnB/+Y3LetlrEqVABMTE1GzZk2pspo1ayI1NRWZmZkyB7dmZ2dL3Uld0ZsoE0JIRff9obs4+zBR3WGohB4vBx21YnE51w4bxrRHN8eaX15JTpUqAZbGihUrsGTJEnWHQQghKpMgyr/tm10NXRjpaKk5GuXREmehYepDCCWZ6Cl8BUMlv7ZKlQAtLS3x9u1bqbK3b9/C0NCw2FsbzZs3D76+vtzzgnPHhBBS1Szs66TUFpI6iUQi7NmzBx8lmTAxMcF33t5lut4nS6VKgG3btsXp06elykJCQtC2bdti1xEKhRAKhcUuJ4QQUvHExcXh48ePMDExgbcKkh+g5gSYlpaGZ8+ecc9jY2MRGRkJU1NT1K5dG/PmzcPr16+xd+9eAMCkSZOwceNGzJkzB9988w3+/fdfHDp0CKdOnVLXSyCEEKICzZo1AwDUqVNHJckPUHMCvHnzJrp06cI9LzhV6e3tjcDAQCQkJCA+Pp5bbm9vj1OnTmHWrFnYsGEDatWqhR07dsg9BpAQQkjFJRKJoKWlBV1dXQD/S4KqotYE+NVXX6GkYYiBgYEy17lz544KoyKEEFLeCq75aWlpwdvbm0uCqkT3AiWEEKJWXIeXjx+Rm5uL3NzcctkvJUBCCCFq83nyU2WHF1koARJCCFELdSY/oJINgyCEkIrsdUom5h29j5SMnHLd79O3aeW6P2VQd/IDKAESQojSnH/0FpeevFfb/q2MZN8QpCKSSCQQi8VqS34AJUBCCFEa8f/fjNrV3hSTO9cr131bGWvDwbLy3ODfxMQEPj4+4PP5akl+ACVAQghROktDbXRxsFB3GBWOSCTC+/fvUb9+fQD5SVCdqBMMIYQQlSu45nfgwAGpO4CpE7UACSGkDGLep+GvW6+QJ2G4/0qk7nAqpMIdXszNzdUdEgBKgIQQUia/nH2Mcw+lZ6nRE9JXa4GK0NuzOPQuEUJIGaRniwEAXRqZo0FNAwg1+fBsTVOuARU7+QGUAAkhRCkGONtgYAsbdYdRYaSnp1fo5AdQAiSEEKICurq6qFOnDgBUyOQHUAIkhFQzEgnD7fiP+JSVp5TtJaeX711fKgsej4f+/fsjIyMDenp66g5HJkqAhJBq5cjtV5hz5J7St8vn85S+zcpGJBLh2rVrcHd3B5/PB4/Hq7DJD6AESAipZhJSsgAApnoC1DJRzq3DLAyE6FjfTCnbqqw+7/ACAD169FBzRF9GCZAQUi31amKJZYOaqjuMKqFwb083Nzd1hyQXuhMMIYSQUqvoQx1KQi1AQkillp0nhkQif/1csQKVSYkqc/IDKAESQiqxP67Gwf/vB/j/SRhIOZJIJNi3b1+lTX4AnQIlhFRiETFJpUp+Who8uNWtofyAqhE+nw8PDw+Ym5tXyuQHUAuQEFIFLOjjCC+32nLX1+DzINTUUGFEVRdjDDxe/pCPevXqYdKkSeDzK2dbihIgIaTSE2ryoSugrzNVE4lEOHr0KPr3748aNfJb0JU1+QF0CpQQQogcCjq8xMfHIzg4GIxV/guv9JOJECK3zaHPikz9o04vktLVHUK1ULi35+DBg7nToJUZJUBCiFyuxCThl7PR6g5DJisj5dzRhRRV2Yc6lIQSICHki7JyxZh/7AEAYICzNfo3t1ZzRP9jqieAs62xusOokqpy8gMoARJC5LA5NAaxSemwMBDip4FNYKitpe6QSDn4559/qmzyAygBEkK+4Nm7T9gS+gwAsLh/Y0p+1Ujfvn0B5N/YuqolP4ASICEEQHTiJ/wd+RpiGT37Lj1JQq6YoauDBXo1sVRDdKQ85eTkQCAQAAB0dHQwdOhQNUekOpQACSFYcuIhrsR8KHa5jpYGlg5oXCV6/pHiFVzzc3V1RZs2bdQdjspRAiSEID07f3b0Hk41UaeGbpHlXRwsUMukaDmpOj7v8HL9+nW0bNmSawlWVZQACSEcz9a26OZYU91hkHImq7dnVU9+AN0JhhBCqrWqPtShJNQCJKQCe5mcgWfv01S+n9SsPJXvg1Q81Tn5AZQACamwUrNy4b7uP2Tnld8Ernw+dXKpTp48eVJtkx9ACZCQCis5LQfZeRLweEBja0OV78/KSAeudqYq3w+pOFq3bg0AaNiwYbVLfgAlQEIqPH2BJk5O76juMEgVkZqaCqFQCKFQCOB/SbA6ok4whBBSTYhEIgQGBmLfvn3Izs5WdzhqRy1AQtSAMfbFa3s54vK79keqvs87vABAdnY21wqsrigBElLOGGPw2n4NEc+Lv/MKIcokq7enoaHqrytXdHQKlJBylitmCiW/9vXNVBgNqeqq+1CHklALkBA1uuLXFYY6Jc+uoCfQKKdoSFVDya9klAAJUSM9oSb0hfRnSFQjOzsb2dnZlPyKQX95hBBSRVlYWMDb2xtCoZCSnwyUAEml8iYlE/OP3cfHjFx1h1JqTMace4Qoi0gkgkgkQu3atQHkJ0EiGyVAUqmEPHqLi9Hv1R2GUhjpaEFHi67vEeUpuOaXlpaGUaNGcUmQyEYJkFQqYkl+68nVzhQTOtVVczRl42htCIEmdcQmylG4wwud8vwySoCkUrI00oa7E81bRwhAvT1Li35+EkJIJUbJr/QoARJCSCWVlpZGya8M6BQoIYRUUjo6OqhZM/9SACU/xZUqAcbHxyMuLg4ZGRkwNzdH48aNq/1NVQkhpLxpaGhgyJAhyMzMhL6+vrrDqXTkToAvXrzAli1bEBQUhFevXkmNZRIIBOjYsSMmTJiAr7/+Gnw+nVklhBBVEIlEiIyMRKdOncDj8aChoUHJr5TkylQzZsxA8+bNERsbi59//hmPHj2CSCRCTk4OEhMTcfr0aXTo0AGLFi1Cs2bNcOPGDVXHTQgh1U5Bh5fQ0FBcunRJ3eFUenK1APX09PD8+XPUqFGjyDILCwt07doVXbt2hb+/P86ePYuXL19W61mGCSFE2Qr39nR2dlZ3SJWeXAlwxYoVcm+wZ8+epQ6GEEJIUTTUQTXoYh0hhFRglPxUR2kJMCoqCnXrVu5bUxFCSEUiFovxxx9/UPJTEaUlwJycHMTFxSlrc4QQUu1paGigS5cuqFGjBiU/FZB7GISvr2+Jy9+/rxp36CeEkIqkcePGcHBwgIYGzRyibHInwA0bNsDZ2RmGhoYyl6elpSktKEIIqa5EIhFOnDiBfv36cS0+Sn6qIfcp0Pr162PWrFm4ePGizMf27dtLFcCmTZtgZ2cHbW1tuLm54fr16yXWDwgIQKNGjaCjowNbW1vMmjULWVlZpdo3IYRUJAUdXmJiYnDixAl1h1PlyZ0AW7VqhVu3bhW7nMfjKTzT9cGDB+Hr6wt/f3/cvn0bzZs3h4eHB969eyez/v79++Hn5wd/f39ERUVh586dOHjwIH788UeF9ksIIRVN4d6e/fr1U3dIVZ7cp0DXrl2L7OzsYpc3b94cEolEoZ2vW7cO48ePx9ixYwEAW7duxalTp7Br1y74+fkVqX/lyhW0b98eXl5eAAA7OzuMGDEC165dU2i/hBBSkdBQB/WQuwVoaWmJOnXqKG3HOTk5uHXrFtzd3f8XDJ8Pd3d3REREyFynXbt2uHXrFnea9Pnz5zh9+jR69+5d7H6ys7ORmpoq9SCEkIqCkp/6qG06pKSkJIjFYm4qjwI1a9bE48ePZa7j5eWFpKQkdOjQAYwx5OXlYdKkSSWeAl2xYgWWLFmi1NhJ+Tp1LwGBV2IhljC8TS3+LAQhldGpU6co+alJpboTTGhoKJYvX47Nmzfj9u3bOHr0KE6dOoWffvqp2HXmzZsHkUjEPV6+fFmOEZOyep2SiR+O3MWNFx9xOz4Fr1MyAQBWRtpqjowQ5ejXrx8aNGhAyU8N1NYCNDMzg4aGBt6+fStV/vbtW1haWspcZ+HChRg9ejS+/fZbAEDTpk2Rnp6OCRMmYP78+TKnYRIKhTRXYSXFGIP/3w+QkSNGy9rGmNS5HgBAqKWBNnVN1RwdIaWXl5cHTc38r18DAwOuXwMpX2prAQoEAri4uODChQtcmUQiwYULF9C2bVuZ62RkZBRJcgXjYxTtgUoqvnMPE3E+6h20NHhY9XUz9GhsiR6NLdG5oTmEmjQuilROIpEIW7Zswd27d9UdSrWnthYgkH93GW9vb7Rq1Qqurq4ICAhAeno61yt0zJgxsLGx4Waj6NevH9atW4cWLVrAzc0Nz549w8KFC9GvXz8aKFrFpGblwj/4IQBgYqd6aFDTQM0REVJ2n3d4uXTpEho3bsy1BEn5K9WRv3TpEnR1ddGqVSuu7ObNm8jIyECnTp3k3o6npyfev3+PRYsWITExEc7Ozjh79izXMSY+Pl6qxbdgwQLweDwsWLAAr1+/hrm5Ofr164dly5aV5mUQBWXnibHvajyS0lTfEeX+axHepmbDroYupnWtr/L9EaJqhXt7jhkzhpKfmvFYKc4d8vl8ODg44NGjR1yZo6Mjnjx5ArFYrNQAlS01NRVGRkYQiUTF3taNyHb2QQIm/Xm7XPe571s3tK9vVq77JETZaKiDcinre7xUPz9iY2OhpaUlVXbhwgXk5uaWOhBS8X3KygMA1DLRQQ8n2R2VlKlpLUNKfqTSo+RXcZUqAcoaEG9tbV3mYEjl0MBCH4v6Oak7DEIqhfv371Pyq6DoBDQhhKhQ+/btAeQP26LkV7HIlQBNTEzA4/Hk2mBycnKZAiKEkMru06dP0NHRgaamJng8Hjp06KDukIgMciXAgIAAFYdBCCFVQ8E1vxo1asDT05N6elZgcr0z3t7eqo6DEEIqvc87vABAZmYmDAxoDGtFVao7wcTExGDBggUYMWIEN3ffmTNn8PDhQ6UGRwghlYWs3p6U/Co2hRPgf//9h6ZNm+LatWs4evQo0tLSAAB3796Fv7+/0gMkhJCKjoY6VE4KJ0A/Pz/8/PPPCAkJgUAg4Mq7du2Kq1evKjU4Un4kEoZcsaTEh1hC91slpDBKfpWXwldn79+/j/379xcpt7CwQFJSklKCIuUrOT0HfX4NQ4IoS92hEFLppKWlIT09nZJfJaRwAjQ2NkZCQgLs7e2lyu/cuQMbGxulBUbKz+OEVIWSX5u6NVQYDSGVi42NDcaMGQN9fX1KfpWMwglw+PDhmDt3Lg4fPgwejweJRILw8HDMnj0bY8aMUUWMpJzUt9DHX5PblVhHk8+DnpC6dZPqTSQSITMzk5u7lH78V04Kf5MtX74cU6dOha2tLcRiMZycnCAWi+Hl5YUFCxaoIkZSTjR4PBjpaH25IiHVWME1v8zMTHh7exc7gTep+BROgAKBANu3b8fChQvx4MEDpKWloUWLFmjQoIEq4iOEkAqjcIcXHR0ddYdEyqDU57Jq164NW1tbAJD7NmlEvY7deYV9V+NRuC9naibN4kHIl1Bvz6qnVAPhd+7ciSZNmkBbWxva2tpo0qQJduzYoezYiJJtuhiDm3EfcavQ4+m7/LGclkbaao6QkIqJkl/VpHALcNGiRVi3bh2mT5+Otm3bAgAiIiIwa9YsxMfHY+nSpUoPkiiH5P/H8f3g0Qj1LfSllvF5PLjam6ojLEIqtNTUVEp+VZTCCXDLli3Yvn07RowYwZX1798fzZo1w/Tp0ykBVgKt7Uwp2REiJx0dHS7hUfKrWhROgLm5uWjVqlWRchcXF+Tl5SklKEIIqSi0tLTg5eWFzMxMGBoaqjscokQKXwMcPXo0tmzZUqR827ZtGDlypFKCIspz+n4C1pyLxppz0fiQnqPucAipFEQiEa5evQrG8i8baGlpUfKrguRqAfr6+nL/5/F42LFjB/755x+0adMGAHDt2jXEx8fTQPgK5t2nLEzZd7tIua5AQw3REFI5FJ7SqOB7jlQ9ciXAO3fuSD13cXEBkD8tEgCYmZnBzMyMpkOqYDKyxQAALQ0eRrrVAQDUqaGLxtb0S5YQWQr39nR0dFR3SESF5EqAFy9eVHUcRIW0NTWwuH9jdYdBSIVGQx2qn1KNAySEkKqEkl/1VKo7wdy8eROHDh1CfHw8cnKkO1YcPXpUKYERQkh5yM3NpeRXTSncAgwKCkK7du0QFRWFY8eOITc3Fw8fPsS///5LHxpCSKWjpaWFdu3awdTUlJJfNVOq2SDWr1+PqVOnwsDAABs2bIC9vT0mTpwIKysrVcRICCEq1apVKzRv3hxaWjQbSnWicAswJiYGffr0AZA/M0R6ejp4PB5mzZqFbdu2KT1AQghRNpFIhIMHDyIjI4Mro+RX/SicAE1MTPDp0ycA+ZNAPnjwAACQkpIi9WEihJCKqKDDy+PHj3HixAl1h0PUSOFToJ06dUJISAiaNm2KoUOHYubMmfj3338REhKCbt26qSJGQghRisK9PXv27KnukIgaKZwAN27ciKysLADA/PnzoaWlhStXruDrr7+mGeEJIRUWDXUghSmcAE1N/zeLAJ/Ph5+fn1IDIoQQZaPkR2SRKwGmpqbKvUG6YSwhpKI5fvw4JT9ShFwJ0NjYGDwer8Q6jDHweDyIxWKlBEYIIcrSv39/BAcHY+DAgZT8CIfuBUoIqZIkEgn4/PyO7gUtP0I+J1cC7Ny5s6rjIIQQpRGJRPjzzz/h7u6ORo0aqTscUkHRzbAJIVVKQYeXpKQknD9/HhKJRN0hkQqKEiAhpMoo3Ntz1KhR3GlQQgqjTwYhpEqgoQ5EUZQACSGVHiU/Uhqlmg8wLy8PoaGhiImJgZeXFwwMDPDmzRsYGhpCX19f2TESGS4/TcK12A8l1knJyC2naAhRr5s3b1LyIwpTOAHGxcWhZ8+eiI+PR3Z2Nrp37w4DAwOsWrUK2dnZ2Lp1qyriJJ8RSxjG772JzFz5xlzqCDRUHBEh6tW1a1cA+dMaUfIj8lI4Ac6cOROtWrXC3bt3UaNGDa580KBBGD9+vFKDI7KJJYxLfiNca0OoWfKZ7G6OFuURFiHlKi0tDbq6uuDz+eDxeHQzfqIwhRNgWFgYrly5AoFAIFVuZ2eH169fKy0wIh+/Xg4w0qF5zEj1UnDNz8bGBoMGDaKenqRUFE6AEolE5u3OXr16BQMDA6UERQghxfm8wwsAZGRkUN8DUioK/2zq0aMHAgICuOc8Hg9paWnw9/dH7969lRkbKeTZuzSEP0vClZgkdYdCiFrI6u1JyY+UlsItwLVr18LDwwNOTk7IysqCl5cXnj59CjMzMxw4cEAVMRIAD16L0Pe3y0XK+SXfo5yQKoOGOhBlUzgB1qpVC3fv3kVQUBDu3buHtLQ0jBs3DiNHjoSOjo4qYiQAXn3MBABoa/FRx1QPANC2Xg0YaNP1P1L1UfIjqqBwAszKyoK2tjZGjRqlinjIFzSxNsKRye3UHQYh5So5ORmpqamU/IhSKZwALSwsMGjQIIwaNQrdunWj3leEEJWzt7fHyJEjYWpqSsmPKI3C2WvPnj3IyMjAgAEDYGNjg++++w43b95URWzVlkTCijwYY+oOi5ByJRKJ8OHD/+52ZG9vT8mPKJXCLcBBgwZh0KBB+PTpE44cOYIDBw6gTZs2qFu3LkaNGoVFixapIs5qY97Rezhw/aW6wyBErQqu+eXm5sLHx0fqphuEKEupz18aGBhg7Nix+Oeff3Dv3j3o6elhyZIlyoytWgp59K7E5a72puUUCSHq8XmHFy0tLWhqluqWxYR8Uak/WVlZWQgODsb+/ftx9uxZ1KxZEz/88IMyY6vWDk5og4Y1pW8swOfz6K4vpEqj3p6kPCmcAM+dO4f9+/fj+PHj0NTUxJAhQ/DPP/+gU6dOqoiv2jLU0YKJnuDLFQmpIij5kfJWqmuAffv2xd69e9G7d29oaVGLhBBSNqmpqZT8SLlTOAG+ffuW7vlZRnEf0rH0xCN8ysorsiwlI0cNERGiXlpaWtyNNCj5kfIiVwJMTU2FoaEhAIAxhtTU1GLrFtQjxTt5LwEXHhff2UWDz4OZvrAcIyJEvXR0dDBq1Cjk5ubSdwgpN3IlQBMTEyQkJMDCwgLGxsbg8YregJIxBh6PJ3OmCCItT5w/pq9jAzOMcK1dZHldcz2YG1ACJFWbSCTC8+fP0aJFCwD5SZBup0jKk1wJ8N9//4WpaX73+4sXL6o0oOqktqkueje1UncYhJS7wlMaFSRBQsqTXAmwc+fO3P/t7e1ha2tbpBXIGMPLlzSAmxBSssK9PevWravukEg1pfBAeHt7e7x//75IeXJyMuzt7RUOYNOmTbCzs4O2tjbc3Nxw/fr1EuunpKRg6tSpsLKyglAoRMOGDXH69GmF90sIKX801IFUJAr3Ai241ldYWloatLW1FdrWwYMH4evri61bt8LNzQ0BAQHw8PBAdHQ0LCwsitTPyclB9+7dYWFhgSNHjsDGxgZxcXEwNjZW9GUQQsoZJT9S0cidAH19fQHkzwC/cOFC6OrqcsvEYjGuXbsGZ2dnhXa+bt06jB8/HmPHjgUAbN26FadOncKuXbvg5+dXpP6uXbuQnJyMK1eucOMP7ezsFNonIaT8ZWdnU/IjFY7cp0Dv3LmDO3fugDGG+/fvc8/v3LmDx48fo3nz5ggMDJR7xzk5Obh16xbc3d3/FwyfD3d3d0RERMhcJzg4GG3btsXUqVNRs2ZNNGnSBMuXL6eep4RUcEKhEC1atKDkRyoUuVuABb0/x44diw0bNpR5rE5SUhLEYjFq1qwpVV6zZk08fvxY5jrPnz/Hv//+i5EjR+L06dN49uwZpkyZgtzcXPj7+8tcJzs7G9nZ2dzzksYwEkJUp2PHjnB1dYVQSEN8SMWgcCeY3bt3q22gqkQigYWFBbZt2wYXFxd4enpi/vz52Lp1a7HrrFixAkZGRtzD1ta2HCP+n6xcMa49/4ArMUmIT85QSwyElCeRSISjR49K/QCl5EcqErlagIMHD0ZgYCAMDQ0xePDgEusePXpUrh2bmZlBQ0MDb9++lSp/+/YtLC0tZa5jZWUFLS0taGhocGWOjo5ITExETk4OBIKiN4+eN28ed/0SyG8BqiMJfn/4Lk7dS5Aq48voTERIVVB4nN+XvjcIUQe5EqCRkRHX81NZ5+4FAgFcXFxw4cIFDBw4EEB+C+/ChQuYNm2azHXat2+P/fv3QyKRgM/Pb7w+efIEVlZWMpMfkP+LsyL86nz1/60+ayNt6GtrQkdLAwNb2Kg5KkKUr3Bvz27duqk7JEJkkisB7t69W+b/y8rX1xfe3t5o1aoVXF1dERAQgPT0dK5X6JgxY2BjY4MVK1YAACZPnoyNGzdi5syZmD59Op4+fYrly5djxowZSotJ1X4a2ATdHGt+uSIhlRANdSCVicLjADMzM8EY44ZBxMXF4dixY3ByckKPHj0U2panpyfev3+PRYsWITExEc7OztzkugAQHx/PtfQAwNbWFufOncOsWbPQrFkz2NjYYObMmZg7d66iL4MQomSU/Ehlw2OMMUVW6NGjBwYPHoxJkyYhJSUFjRo1gkAgQFJSEtatW4fJkyerKlalSE1NhZGREUQiUbl25hmw8TLuvhJhp3cragGSKocxhl27duHVq1eU/IjKKet7XOFeoLdv30bHjh0BAEeOHIGlpSXi4uKwd+9e/Prrr6UOhBBSefF4PPTr1w+1atWi5EcqDYVPgWZkZHAT4v7zzz8YPHgw+Hw+2rRpg7i4OKUHSAipuD6/NaKFhQW++eYbmbdKJKQiUrgFWL9+fRw/fhwvX77EuXPnuOt+7969o4ksCalGRCIRfv/9d7x48YIro+RHKhOFE+CiRYswe/Zs2NnZwdXVFW3btgWQ3xqkOb0IqR4KOry8ffsWZ8+ehYJdCQipEBQ+BTpkyBB06NABCQkJaN68OVferVs3DBo0SKnBEUIqnsK9PUeMGEEtP1IpKZwAAcDS0hKWlpZ49eoVAKBWrVpwdXVVamCEkIqHhjqQqkThU6ASiQRLly6FkZER6tSpgzp16sDY2Bg//fQTJBKJKmIkhFQAlPxIVaNwC3D+/PnYuXMnVq5cifbt2wMALl++jMWLFyMrKwvLli1TepCEEPW7cuUKJT9SpSicAPfs2YMdO3agf//+XFnBXVmmTJlCCZCQKqqgx3e7du0o+ZEqQeEEmJycDAcHhyLlDg4OSE5OVkpQhJCKISMjAzo6OuDxeNDQ0ECvXr3UHRIhSqPwNcDmzZtj48aNRco3btwo1SuUEFK5iUQi7NixA6dPn6ZhDqRKUrgF+Msvv6BPnz44f/48NwYwIiICL1++xOnTp5UeICGk/H3e4SUmJgaZmZncDfAJqSoUbgF27twZT548weDBg5GSkoKUlBQMHjwY0dHR3D1CCSGVl6zenpT8SFWkUAvwxYsXCAkJQU5ODoYPH44mTZqoKi5CiBrQUAdSncidAC9evIi+ffsiMzMzf0VNTezatQujRo1SWXCEkPJDyY9UN3KfAl24cCG6d++O169f48OHDxg/fjzmzJmjytgqvaS0bETEfEBEzAd8ys5TdziElCgxMREpKSmU/Ei1IfeEuMbGxrhy5QqcnJwA5HePNjQ0xNu3b1GjRg2VBqlM5TUhbnaeGG7LLyAlI1eqfLdPa3RxsFDZfgkpi+joaFhaWlLyIxWasr7H5T4FmpqaCjMzM+65rq4udHR0IBKJKlUCLC9pWXlc8qtvoQ8AsDLSRis7E3WGRYgUkUgEAFzCa9SokTrDIaRcKdQJ5ty5c1K/DCUSCS5cuIAHDx5wZZ/fIYbkO+/bWd0hEFJEwTU/AHTKk1RLCiVAb2/vImUTJ07k/s/j8SAWi8seFSFEpQp3eCGkOpI7AdJMD4RUDdTbk5B8pZoPkPzPoZsvseD4A+Tk0Q8EUvFR8iPkf+QaBnH16lW5N5iRkYGHDx+WOqDKJjT6XYnJr1UdOr1EKgZKfoRIk6sFOHr0aNStWxfffvstevfuDT09vSJ1Hj16hD///BO7d+/GqlWr0LhxY6UHW5HN7emAoa1qFSk31RWoIRpCiuLz+dDQ0KDkR8j/kysBPnr0CFu2bMGCBQvg5eWFhg0bwtraGtra2vj48SMeP36MtLQ0DBo0CP/88w+aNm2q6rgrHH2hBsz0heoOg5BiGRgYYMyYMZBIJJT8CIGcCVBLSwszZszAjBkzcPPmTVy+fBlxcXHIzMxE8+bNMWvWLHTp0gWmpqaqjpcQogCRSIRXr15xZ2QMDAzUHBEhFYfCnWBatWqFVq1aqSIWQogSfX7ND0C1uyxByJcoPB0SIaTiK9zhpVatotenCanuKAESUsVQb09C5EMJkJAqhJIfIfKjBEhIFZGZmUnJjxAFlOlOMFlZWdDW1lZWLJXGrbiPuPw0CQDw9G2amqMhJJ+2tjYcHBzw+PFjSn6EyEHhBCiRSLBs2TJs3boVb9++xZMnT1C3bl0sXLgQdnZ2GDdunCrirFAm/nELSWnZUmXaWhpqioaQfDweD927d0fHjh2ho6Oj7nAIqfAUPgX6888/IzAwEL/88gsEgv/d5aRJkybYsWOHUoOrqD5l5c/zN9DZGiPdamNG1/ro1dRKzVGR6kgkEuHkyZPIy8sDkJ8EKfkRIh+FW4B79+7Ftm3b0K1bN0yaNIkrb968OR4/fqzU4Cq62R6NUMtEV91hkGqq8Di/vn37qjkiQioXhVuAr1+/Rv369YuUSyQS5ObmKiUoQkjJCvf27Nixo7pDIqTSUbgF6OTkhLCwMNSpU0eq/MiRI2jRooXSAqtoXiZn4E1KJgBAwpiaoyHVGQ11IEQ5FE6AixYtgre3N16/fg2JRIKjR48iOjoae/fuxcmTJ1URo9rFf8hA5zUXUTjv8Xk89QREqi1KfoQoj8KnQAcMGIATJ07g/Pnz0NPTw6JFixAVFYUTJ06ge/fuqohR7V59zABjgECDj3rmeqhnrofBLW1gZVT9hoAQ9WGMISgoiJIfIUpSqnGAHTt2REhIiLJjqfDszfRwblYndYdBqikej4c+ffrg1KlTGD58OCU/QspI4RZg3bp18eHDhyLlKSkpqFu3rlKCIoT8D/vs3HutWrUwYcIESn6EKIHCCfDFixcQi8VFyrOzs/H69WulBEUIyScSibBjxw4kJCRwZTy69kyIUsh9CjQ4OJj7/7lz56R+gYrFYly4cAF2dnZKDa68rTgThe2XnqNwH0/q9EnU4fMOLydPnsS3335LyY8QJZI7AQ4cOBBA/q9Pb29vqWVaWlqws7PD2rVrlRpceQt5+BaSEpKdi51J+QVDqrXCvT2HDRtGyY8QJZM7AUokEgCAvb09bty4ATMzM5UFpW6/j3ZBi9rGUmV8Hg9m+kL1BESqFRrqQEj5ULgXaGxsrCriqFBMdAWwMKAhDqT8UfIjpPyUahhEeno6/vvvP8THxyMnJ0dq2YwZM5QSGCHVUWhoKCU/QsqJwgnwzp076N27NzIyMpCeng5TU1MkJSVBV1cXFhYWlAAJKYPevXsDAL766itKfoSomMLDIGbNmoV+/frh48eP0NHRwdWrVxEXFwcXFxesWbNGFTESUqVlZWVxY/20tLQwYMAASn6ElAOFE2BkZCS+//578Pl8aGhoIDs7G7a2tvjll1/w448/qiJGQqoskUiEbdu24eLFi1ID3gkhqqdwAtTS0gKfn7+ahYUF4uPjAQBGRkZ4+fKlcqMjpAr7vMPLgwcPkJ2dre6QCKlWFL4G2KJFC9y4cQMNGjRA586dsWjRIiQlJeGPP/5AkyZNVBEjIVWOrN6e2trU85iQ8qRwC3D58uWwsrICACxbtgwmJiaYPHky3r9/j99//13pARJS1dBQB0IqBoVbgK1ateL+b2FhgbNnzyo1IEKqMkp+hFQcCrcAi3P79m307dtXWZsjpEqKi4uj5EdIBaFQC/DcuXMICQmBQCDAt99+i7p16+Lx48fw8/PDiRMn4OHhoao4CakSmjVrBgCoU6cOJT9C1EzuBLhz506MHz8epqam+PjxI3bs2IF169Zh+vTp8PT0xIMHD+Do6KjKWAmplEQiEbS0tKCrqwvgf0mQEKJecp8C3bBhA1atWoWkpCQcOnQISUlJ2Lx5M+7fv4+tW7dS8iNEhoJrfnv27EFGRoa6wyGEfEbuBBgTE4OhQ4cCAAYPHgxNTU2sXr0atWrVUllwhFRmn3d4yc3NRW5urrpDIoR8Ru4EmJmZyZ3C4fF4EAqF3HAIQog06u1JSMWnUCeYHTt2QF9fHwCQl5eHwMDAIvMC0s2wSXVHyY+QykHuBFi7dm1s376de25paYk//vhDqg6Px6MESKo1Sn6EVB5ynwJ98eIFYmNjS3w8f/68VEFs2rQJdnZ20NbWhpubG65fvy7XekFBQeDxeBg4cGCp9kuIskkkEkgkEkp+hFQCpZoQV5kOHjwIX19fbN26FW5ubggICICHhweio6NhYWFR7HovXrzA7Nmz0bFjx3KMlpCSFSQ+Pp9PyY+QCk5pd4IprXXr1mH8+PEYO3YsnJycsHXrVujq6mLXrl3FriMWizFy5EgsWbIEdevWLcdoCSlKJBLh2bNn3HMTExNKfoRUAmpNgDk5Obh16xbc3d25Mj6fD3d3d0RERBS73tKlS2FhYYFx48Z9cR/Z2dlITU2VehCiLAXX/A4cOCCVBAkhFZ9aE2BSUhLEYjFq1qwpVV6zZk0kJibKXOfy5cvYuXOnVIeckqxYsQJGRkbcw9bWtsxxEwJId3gxMjKCubm5ukMihChA7adAFfHp0yeMHj0a27dvLzL8ojjz5s2DSCTiHjRpL1EG6u1JSOVXqk4wMTEx2L17N2JiYrBhwwZYWFjgzJkzqF27Nho3biz3dszMzKChoYG3b99Klb99+xaWlpYy9/vixQv069ePK5NIJPkvRFMT0dHRqFevntQ6QqEQQqFQkZdHSIko+RFSNSjcAvzvv//QtGlTXLt2DUePHkVaWhoA4O7du/D391doWwKBAC4uLrhw4QJXJpFIcOHCBbRt27ZIfQcHB9y/fx+RkZHco3///ujSpQsiIyPp9CZRufT0dEp+hFQRCrcA/fz88PPPP8PX1xcGBgZcedeuXbFx40aFA/D19YW3tzdatWoFV1dXBAQEID09HWPHjgUAjBkzBjY2NlixYgW0tbXRpEkTqfWNjY0BoEg5Iaqgq6uLOnXqAAAlP0IqOYUT4P3797F///4i5RYWFkhKSlI4AE9PT7x//x6LFi1CYmIinJ2dcfbsWa5jTHx8PPj8SnWpklRhPB4P/fv3R0ZGBvT09NQdDiGkDBROgMbGxkhISIC9vb1U+Z07d2BjY1OqIKZNm4Zp06bJXBYaGlriuoGBgaXaJyHyEolEuHbtGtzd3cHn88Hj8Sj5EVIFKNy0Gj58OObOnYvExETweDxIJBKEh4dj9uzZGDNmjCpiJERtCjq8RERE4Pz58+oOhxCiRAonwOXLl8PBwQG2trZIS0uDk5MTOnXqhHbt2mHBggWqiJEQtSjc29PNzU3dIRFClEjhU6ACgQDbt2/HwoUL8eDBA6SlpaFFixZo0KCBKuIjRC1oqAMhVZ/CCfDy5cvo0KEDateujdq1a6siJkLUipIfIdWDwqdAu3btCnt7e/z444949OiRKmIiRG0kEgn27dtHyY+QakDhBPjmzRt8//33+O+//9CkSRM4Oztj9erVePXqlSriI6Rc8fl8eHh4wNzcnJIfIVWcwgnQzMwM06ZNQ3h4OGJiYjB06FDs2bMHdnZ26Nq1qypiJETlGGPc/+vVq4dJkyZR8iOkiivTCHN7e3v4+flh5cqVaNq0Kf777z9lxUVIuRGJRAgMDJS6kQPdfIGQqq/Uf+Xh4eGYMmUKrKys4OXlhSZNmuDUqVPKjI0QlSvo8BIfH48TJ05ItQQJIVWbwr1A582bh6CgILx58wbdu3fHhg0bMGDAAOjq6qoiPkJUpnBvz8GDB4PH46k7LEJIOVE4AV66dAk//PADhg0bJvecfIRUNDTUgRCicAIMDw9XRRyElBtKfoQQQM4EGBwcjF69ekFLSwvBwcEl1u3fv79SAiNEVf755x9KfoQQ+RLgwIEDkZiYCAsLCwwcOLDYejweD2KxWFmxEaISffv2BQD06NGDkh8h1ZhcCVAikcj8PyGVRU5ODgQCAQBAR0cHQ4cOVXNEhBB1U3gYxN69e5GdnV2kPCcnB3v37lVKUIQok0gkwtatW3H16lV1h0IIqUAUToBjx46FSCQqUv7p0yeMHTtWKUERoiyfd3i5fv06cnJy1B0SIaSCUDgBMsZkjpV69eoVXU8hFYqs3p4Fp0EJIUTuYRAtWrQAj8cDj8dDt27doKn5v1XFYjFiY2PRs2dPlQRJiKJoqAMh5EvkToAFvT8jIyPh4eEBfX19bplAIICdnR2+/vprpQdIiKIo+RFC5CF3AvT39wcA2NnZwdPTE9ra2ioLipCyePr0KSU/QsgXKXwnGG9vb1XEQYjStGrVCgDQoEEDSn6EkGLJlQBNTU3x5MkTmJmZwcTEpMQbBicnJystOELklZqaCqFQCKFQCOB/SZAQQoojVwJcv349DAwMuP/THfNJRVJwzU9fXx8jR47kkiAhhJRErgT4+WlPHx8fVcVCiMI+7/ACANnZ2ZQACSFyUXgc4O3bt3H//n3u+d9//42BAwfixx9/pEHGpFzJ6u1paGio7rAIIZWEwglw4sSJePLkCQDg+fPn8PT0hK6uLg4fPow5c+YoPUBCZKGhDoSQslI4AT558gTOzs4AgMOHD6Nz587Yv38/AgMD8ddffyk7PkKKoORHCFGGUt0KrWBGiPPnz6N3794AAFtbWyQlJSk3OkJkyM7ORnZ2NiU/QkiZKDwOsFWrVvj555/h7u6O//77D1u2bAEAxMbGombNmkoPkJDCLCws4O3tDaFQSMmPEFJqCrcAAwICcPv2bUybNg3z589H/fr1AQBHjhxBu3btlB4gIUD+ac/4+HjuuYWFBSU/QkiZKNwCbNasmVQv0AKrV6+GhoaGUoIi5HMF1/zS0tIwatQo1K5dW90hEUKqAIUTYIFbt24hKioKAODk5ISWLVsqLShCChTu8EKtPkKIsiicAN+9ewdPT0/8999/MDY2BgCkpKSgS5cuCAoKgrm5ubJjJNUU9fYkhKiSwtcAp0+fjrS0NDx8+BDJyclITk7GgwcPkJqaihkzZqgiRlINUfIjhKiawi3As2fP4vz583B0dOTKnJycsGnTJvTo0UOpwZHqKS0tjZIfIUTlFE6AEokEWlpaRcq1tLS48YGElIWOjg43pIaSHyFEVRROgF27dsXMmTNx4MABWFtbAwBev36NWbNmoVu3bkoPkFQ/GhoaGDJkCDIzM6Gvr6/ucAghVZTC1wA3btyI1NRU2NnZoV69eqhXrx7s7e2RmpqK3377TRUxkmpAJBIhNDQUjDEA+UmQkh8hRJUUbgHa2tri9u3buHDhAjcMwtHREe7u7koPjlQPhac0+uqrr9QbECGkWlAoAR48eBDBwcHIyclBt27dMH36dFXFRaqJwr09W7Rooe6QCCHVhNwJcMuWLZg6dSoaNGgAHR0dHD16FDExMVi9erUq4yNVGA11IISok9zXADdu3Ah/f39ER0cjMjISe/bswebNm1UZW7m4GP0Om0OfYXPoM3zMoAl9ywslP0KIuvFYQa+DL9DR0UFUVBTs7OwA5A+H0NHRwYsXL2BlZaXKGJUqNTUVRkZGEIlEyOEL0XrZeRQ+AsHT2qNZLWO1xFcdiMVibNmyBR8+fKDkRwhR2Off44aGhqXejtynQLOzs6Gnp8c95/P5EAgEyMzMLPXO1S0tKw+MAZp8Hga3tAEA1KmhhybW9GWsShoaGujSpQsuXryI0aNHU/IjhKiFQp1gFi5cCF1dXe55Tk4Oli1bJvUFtm7dOuVFV050tDTwy5Dm6g6jWmncuDEcHBxoBhFCiNrInQA7deqE6OhoqbJ27drh+fPn3HMej6e8yEiVIhKJEBwcjP79+3M/mCj5EULUSe4EGBoaqsIwSFX2eYeXEydOYNSoUeoOiRBCFL8TDCGKKNzbs1+/fuoOiRBCAFACJCpEQx0IIRUZJUCiEpT8CCEVHSVAohKnTp2i5EcIqdAoARKV6N+/Pxo0aEDJjxBSYZUqAYaFhWHUqFFo27YtXr9+DQD4448/cPnyZaUGRyqXvLw87v/6+vrw8vKi5EcIqbAUToB//fUXPDw8oKOjgzt37iA7OxtA/jWf5cuXKz1AUjmIRCJs2bIFd+/eVXcohBAiF4UT4M8//4ytW7di+/bt0NLS4srbt2+P27dvKzU4UjkUdHhJTk7GpUuXpFqChBBSUSmcAKOjo9GpU6ci5UZGRkhJSVFGTKQSKdzbc8yYMdDUVHieZUIIKXcKJ0BLS0s8e/asSPnly5dRt25dpQRFKgca6kAIqcwUToDjx4/HzJkzce3aNfB4PLx58wb79u3D7NmzMXnyZFXESCogSn6EkMpO4XNVfn5+kEgk6NatGzIyMtCpUycIhULMnj0b06dPV0WMpAK6f/8+JT9CSKWmcALk8XiYP38+fvjhBzx79gxpaWlwcnKCvr6+KuIjFVT79u0BAE2bNqXkRwiplEo9EF4gEMDJyQmurq5lTn6bNm2CnZ0dtLW14ebmhuvXrxdbd/v27ejYsSNMTExgYmICd3f3EusT5fn06RPXw5PH46FDhw6U/AghlZbCLcAuXbqUOO/fv//+q9D2Dh48CF9fX2zduhVubm4ICAiAh4cHoqOjYWFhUaR+aGgoRowYgXbt2kFbWxurVq1Cjx498PDhQ9jY2Cj6coicCq751ahRA56entTTkxBS6SncAnR2dkbz5s25h5OTE3JycnD79m00bdpU4QDWrVuH8ePHY+zYsXBycsLWrVuhq6uLXbt2yay/b98+TJkyBc7OznBwcMCOHTsgkUhw4cIFhfdN5PN5h5cPHz4gMzNT3SERQkiZKfwzfv369TLLFy9ejLS0NIW2lZOTg1u3bmHevHlcGZ/Ph7u7OyIiIuTaRkZGBnJzc2FqaqrQvol8ZPX2NDAwUHdYhBBSZkq7GfaoUaOKbbUVJykpCWKxGDVr1pQqr1mzJhITE+Xaxty5c2FtbQ13d3eZy7Ozs5Gamir1IPKhoQ6EkKpMaQkwIiIC2traytqcXFauXImgoCAcO3as2H2vWLECRkZG3MPW1rZcY6ysKPkRQqo6hU+BDh48WOo5YwwJCQm4efMmFi5cqNC2zMzMoKGhgbdv30qVv337FpaWliWuu2bNGqxcuRLnz59Hs2bNiq03b948+Pr6cs9TU1MpCcohLS0N6enplPwIIVWWwgmw8Bchn89Ho0aNsHTpUvTo0UOhbQkEAri4uODChQsYOHAgAHAdWqZNm1bser/88guWLVuGc+fOoVWrViXuQygUQigUKhQXAWxsbDBmzBjo6+tT8iOEVEkKJUCxWIyxY8eiadOmMDExUUoAvr6+8Pb2RqtWreDq6oqAgACkp6dj7NixAIAxY8bAxsYGK1asAACsWrUKixYtwv79+2FnZ8ddK9TX16fB+GUkEomQkZEBKysrAKBhJYSQKk2hBKihoYEePXogKipKaQnQ09MT79+/x6JFi5CYmAhnZ2ecPXuW6xgTHx8PPv9/lyq3bNmCnJwcDBkyRGo7/v7+WLx4sVJiqo4KrvllZmZizJgxXBIkhJCqSuFToE2aNMHz589hb2+vtCCmTZtW7CnP0NBQqecvXrxQ2n5JvsIdXnR1ddUdEiGEqFypJsSdPXs2Tp48iYSEBBpiUMlRb09CSHUldwtw6dKl+P7779G7d28AQP/+/aVuicYYA4/Hg1gsVn6URCUo+RFCqjO5E+CSJUswadIkXLx4UZXxkHKSmppKyY8QUq3JnQAZYwCAzp07qywYUn50dHS4hEfJjxBSHSnUCaakWSBI5aKlpQUvLy9kZmbC0NBQ3eEQQki5UygBNmzY8ItJMDk5uUwBEdURiUR49OgR2rRpAx6PBy0tLWhpaak7LEIIUQuFEuCSJUvoVFkl9XmHFwBo27atmiMihBD1UigBDh8+XOYktaRiK9zb08nJSd0hEUKI2sk9DpCu/1VONNSBEEJkkzsBFvQCJZUHJT9CCCme3KdAJRKJKuMgSpabm0vJjxBCSqC0CXFJxaKlpYX27dvD1NSUkh8hhMig8M2wSeXh4uKCZs2a0VAHQgiRgVqAVYhIJMLBgweRkZHBlVHyI4QQ2agFWEUUHufn6emp5ogIIaRioxZgFVC4t2fPnj3VHRIhhFR41AKs5CrSUAexWIzc3Fy17JsQUnVoaWlBQ0ND5fuhBFiJVZTkxxhDYmIiUlJSyn3fhJCqydjYGJaWliq9CQslwErs77//VnvyA8AlPwsLC+jq6tJdgwghpcYYQ0ZGBt69ewcAsLKyUtm+KAFWYv369cOJEycwYMAAtZ72LEh+NWrUUEsMhJCqRUdHBwDw7t07WFhYqOx0KCXASkYikYDPz++7ZGJigjFjxqg1noJrfrq6umqNgxBStRR8p+Tm5qosAVIv0EpEJBJhy5YtiI6OVncoRdBpT0KIMpXHdwolwEqioMNLUlISzp8/T/dmrSTs7OwQEBBQ6vUDAwNhbGystHiqkrIeW0WMHj0ay5cvL5d9VQdnz56Fs7Oz2r/HKAFWAoV7e44aNYo7DUpKz8fHBwMHDlTpPm7cuIEJEybIVVfWF7qnpyeePHlS6v0HBgaCx+OBx+OBz+fDysoKnp6eiI+PL/U2KwpFjm1Z3L17F6dPn8aMGTOKLDtw4AA0NDQwderUIstK+vHC4/Fw/PhxqbK//voLX331FYyMjKCvr49mzZph6dKlSE5OVsbLkCk5ORkjR46EoaEhjI2NMW7cOKSlpZW4TmJiIkaPHg1LS0vo6emhZcuW+Ouvv6Tq2NnZcZ+7gsfKlSu55T179oSWlhb27dunktclL/oWreAqylAHUjrm5uZluj6qo6NT5kmoDQ0NkZCQgNevX+Ovv/5CdHQ0hg4dWqZtykPVY0LLemzl9dtvv2Ho0KHQ19cvsmznzp2YM2cODhw4gKysrFLvY/78+fD09ETr1q1x5swZPHjwAGvXrsXdu3fxxx9/lCX8Eo0cORIPHz5ESEgITp48iUuXLn3xR8WYMWMQHR2N4OBg3L9/H4MHD8awYcNw584dqXpLly5FQkIC95g+fbrUch8fH/z6669Kf00KYdWMSCRiAJhIJGKx79NYnbknWZNFZ9UdlkwpKSlsw4YNbPHixWzDhg0sJSVF3SEVkZmZyR49esQyMzPVHYrCvL292YABA4pdHhoaylq3bs0EAgGztLRkc+fOZbm5udzy1NRU5uXlxXR1dZmlpSVbt24d69y5M5s5cyZXp06dOmz9+vWMMcYkEgnz9/dntra2TCAQMCsrKzZ9+nTGGGOdO3dmAKQejDG2e/duZmRkJBVXcHAwa9WqFRMKhaxGjRps4MCBxb4GWev/+uuv3N9AgePHj7MWLVowoVDI7O3t2eLFi6Vea1RUFGvfvj0TCoXM0dGRhYSEMADs2LFjjDHGYmNjGQAWFBTEOnXqxIRCIdu9ezdjjLHt27czBwcHJhQKWaNGjdimTZu47WZnZ7OpU6cyS0tLJhQKWe3atdny5cu/eLwKH1vGGIuLi2P9+/dnenp6zMDAgA0dOpQlJiZyy/39/Vnz5s3Z3r17WZ06dZihoSHz9PRkqampxR6/vLw8ZmRkxE6ePFlk2fPnz5mOjg5LSUlhbm5ubN++fV889gU+P3bXrl1jAFhAQIDMuh8/fiw2vrJ49OgRA8Bu3LjBlZ05c4bxeDz2+vXrYtfT09Nje/fulSozNTVl27dv554Xfm9kiYuLYwDYs2fPZC4v6bvl8+/xsqAWYAV28+bNStnyY4whIydPLQ+mpImbX79+jd69e6N169a4e/cutmzZgp07d+Lnn3/m6vj6+iI8PBzBwcEICQlBWFgYbt++Xew2//rrL6xfvx6///47nj59iuPHj6Np06YAgKNHj6JWrVpSv5plOXXqFAYNGoTevXvjzp07uHDhAlxdXeV+Xe/evcOxY8egoaHB9awLCwvDmDFjMHPmTDx69Ai///47AgMDsWzZMgD5Q10GDhwIXV1dXLt2Ddu2bcP8+fNlbt/Pzw8zZ85EVFQUPDw8sG/fPixatAjLli1DVFQUli9fjoULF2LPnj0AgF9//RXBwcE4dOgQoqOjsW/fPtjZ2X3xeBUmkUgwYMAAJCcn47///kNISAieP39e5J64MTExOH78OE6ePImTJ0/iv//+kzo1V9i9e/cgEonQqlWrIst2796NPn36wMjICKNGjcLOnTtLPvjF2LdvH/T19TFlyhSZy0u6Bty4cWPo6+sX++jVq1ex60ZERMDY2Fjqtbm7u4PP5+PatWvFrteuXTscPHgQycnJkEgkCAoKQlZWFr766iupeitXrkSNGjXQokULrF69Gnl5eVLLa9eujZo1ayIsLKzYfakaDYOowLp27QoAaNWqVaVJfgCQmSuG06Jzatn3o6Ue0BWU/WO9efNm2NraYuPGjeDxeHBwcMCbN28wd+5cLFq0COnp6dizZw/279+Pbt26Acj/QrS2ti52m/Hx8bC0tIS7uzu0tLRQu3ZtLnmZmppCQ0MDBgYGsLS0LHYby5Ytw/Dhw7FkyRKurHnz5iW+FpFIBH19fW6AMQDMmDEDenp6AIAlS5bAz88P3t7eAIC6devip59+wpw5c+Dv74+QkBDExMQgNDSUi23ZsmXo3r17kX199913GDx4MPfc398fa9eu5crs7e25JOvt7Y34+Hg0aNAAHTp0AI/HQ506deQ6XoVduHAB9+/fR2xsLGxtbQEAe/fuRePGjXHjxg20bt0aQH6iDAwMhIGBAYD8zi0XLlzgkn1hcXFx0NDQKHIaumA7v/32GwBg+PDh+P777xEbGwt7e/ti3wtZnj59irp165Zq5pbTp0+XeKq5YDydLImJiUVel6amJkxNTZGYmFjseocOHYKnpydq1KgBTU1N6Orq4tixY6hfvz5XZ8aMGWjZsiVMTU1x5coVzJs3DwkJCVi3bp3UtqytrREXF/ell6kylAArmLS0NOjq6oLP54PH43FfrqR8RUVFoW3btlJdsdu3b4+0tDS8evUKHz9+RG5urtQXspGRERo1alTsNocOHYqAgADUrVsXPXv2RO/evdGvXz9oasr/ZxgZGYnx48cr9FoMDAxw+/Zt5Obm4syZM9i3b5/UF/7du3cRHh4uVSYWi5GVlYWMjAxER0fD1tZWKjEXl4g+b02kp6cjJiYG48aNk4o5Ly+P+0Hn4+OD7t27o1GjRujZsyf69u2LHj16AFDseEVFRcHW1pZLfgDg5OQEY2NjREVFcQnQzs6OS35A/l1GCu44IktmZiaEQmGRLvkhISFIT09H7969AQBmZmbo3r07du3ahZ9++qnY7clSlrMWn/9gKC8LFy5ESkoKzp8/DzMzMxw/fhzDhg1DWFgY10L39fXl6jdr1gwCgQATJ07EihUrIBQKuWU6OjpS07eVN0qAFUhBhxdra2sMHjy40vb01NHSwKOlHmrbd0Vla2uL6OhonD9/HiEhIZgyZQpWr16N//77T+5f/yX9oi8On8/nfp07OjoiJiYGkydP5jpXpKWlYcmSJVIttwLa2toK7augVVmwXQDYvn073NzcpOoVnH5t2bIlYmNjcebMGZw/fx7Dhg2Du7s7jhw5opTjVVjh9Xg8Xold8c3MzJCRkYGcnBwIBAKufOfOnUhOTpZ6PyQSCe7du4clS5aAz+fD0NAQ6enpUjevAMDdM7fgR0DDhg1x+fJl5ObmKvy6GjduXGILqmPHjjhz5ozMZZaWlkWSf15eHpKTk4s9CxETE4ONGzfiwYMHaNy4MYD8MxBhYWHYtGkTtm7dKnM9Nzc35OXl4cWLF1I/EpOTk2Fubl7ia1QlSoAVROH5/DIyMmT2OqsMeDyeUk5DqpOjoyP++usvMMa4X//h4eEwMDBArVq1YGJiAi0tLdy4cQO1a9cGkP8ePnnyBJ06dSp2uzo6OujXrx/69euHqVOnwsHBAffv30fLli0hEAggFotLjKtZs2a4cOECxo4dW+rX5ufnh3r16mHWrFlo2bIlWrZsiejoaKlTWJ9r1KgRXr58ibdv36JmzZoA8ocgfEnNmjVhbW2N58+fY+TIkcXWMzQ0hKenJzw9PTFkyBD07NkTycnJMDU1LfF4fc7R0REvX77Ey5cvuVbgo0ePkJKSAicnJ3kPTRHOzs7ctgr+/+HDB/z9998ICgrikgCQ32ru0KED/vnnH/Ts2RONGjVCXl4eIiMjpeItuE7csGFDAICXlxd+/fVXbN68GTNnziwSQ0pKSrHXActyCrRt27ZISUnBrVu34OLiAgD4999/IZFIivxgKVDQWiv841xDQ6PEHxKRkZHg8/lSp1yzsrIQExODFi1aFLueqlXub6kqQtZQh8qa/CobkUiEyMhIqbIaNWpgypQpCAgIwPTp0zFt2jRER0fD398fvr6+4PP5MDAwgLe3N3744QeYmprCwsIC/v7+3KlrWQIDAyEWi+Hm5gZdXV38+eef0NHR4U5j2dnZ4dKlSxg+fDiEQiHMzMyKbMPf3x/dunVDvXr1MHz4cOTl5eH06dOYO3eu3K/Z1tYWgwYNwqJFi3Dy5EksWrQIffv2Re3atTFkyBDw+XzcvXsXDx48wM8//4zu3bujXr168Pb2xi+//IJPnz5hwYIFAL58t44lS5ZgxowZMDIyQs+ePZGdnc117vL19cW6detgZWWFFi1agM/n4/Dhw7C0tISxsfEXj9fn3N3d0bRpU4wcORIBAQHIy8vDlClT0LlzZ5kdWORlbm6Oli1b4vLly1wC/OOPP1CjRg0MGzasyOvv3bs3du7ciZ49e6Jx48bo0aMHvvnmG6xduxZ169ZFdHQ0vvvuO3h6esLGxgZAfutozpw5+P777/H69WsMGjQI1tbWePbsGbZu3YoOHTrITIxA2U6BOjo6omfPnhg/fjy2bt2K3NxcTJs2DcOHD+euZb9+/RrdunXD3r174erqCgcHB9SvXx8TJ07EmjVrUKNGDRw/fpwbRgHkd665du0aunTpAgMDA0RERGDWrFkYNWoUTExMuP1fvXoVQqEQbdu2LfVrKLMy9SGthCraMIjKMNShJJV9GAQKDT0AwMaNG8cYK90wCFdXV+bn58fV+bw7+LFjx5ibmxszNDRkenp6rE2bNuz8+fNc3YiICNasWTMmFApLHAbx119/MWdnZyYQCJiZmRkbPHhwsa+xuK74ERERDAC7du0aY4yxs2fPsnbt2jEdHR1maGjIXF1d2bZt27j6BcMgBAIBc3BwYCdOnGAA2Nmz+X87BcMg7ty5U2Rf+/bt4+I1MTFhnTp1YkePHmWMMbZt2zbm7OzM9PT0mKGhIevWrRu7ffu2XMertMMgPrd+/XpWp06dYo8fY4xt3ryZtWnThnvetGlTNmXKFJl1Dx48yAQCAXv//j1jLH8Iw4wZM1i9evWYjo4Oa9CgAZszZw779OmTzHU7derEDAwMmJ6eHmvWrBlbunSpyoZBMMbYhw8f2IgRI5i+vj4zNDRkY8eOlYqt4H29ePEiV/bkyRM2ePBgZmFhwXR1dVmzZs2khkXcunWLubm5MSMjI6atrc0cHR3Z8uXLWVZWltS+J0yYwCZOnFhsbOUxDILHmJL6jVcSqampMDIygkgkQnKOBr5aEwoDoSbuLyn/a1ZVYZB7VlYW1/NN0etFVU16ejpsbGywdu1ajBs3Tt3hqFR4eDg6dOiAZ8+eoV69euoOR6UyMzPRqFEjHDx4UL2tlSokKSkJjRo1ws2bN4vtNVvSd8vn3+OGhoaljoNOgapRcnIyUlNTK23yq+7u3LmDx48fw9XVFSKRCEuXLgUADBgwQM2RKd+xY8egr6+PBg0a4NmzZ5g5cybat29f5ZMfkH8dbe/evUhKSlJ3KFXGixcvsHnzZoWHjCgbJUA1sre3x8iRI2FqakrJr5Jas2YNoqOjIRAI4OLigrCwMJnX7iq7T58+Ye7cuYiPj4eZmRnc3d2xdu1adYdVbgoP8iZl06pVqzJdm1UWSoDlTCQSITc3l/uSVPcvIFJ6LVq0wK1bt9QdRrkYM2aM2ueeJETZKudAs0qq4JpfwbRGhBBC1IcSYDn5vMOLlpZWqQfyEkIIUQ5KgOWgKvT2JISQqoYSoIpR8iOEkIqJEqAKUfIjhJCKi3qBqpBQKOTuxUfJjxBCKhZqAaqQtrY2Ro8eDR8fH0p+pAgej4fjx4+rO4xK48KFC3B0dPziDcOrAz8/P0yfPl3dYVR6lACVTCQS4c6dO9xzbW3tMt2qh6iOj48PeDweeDwetLS0YG9vjzlz5iArK0vdoREZ5syZgwULFnBTKRXIzMyEqakpzMzMkJ2dXWS94n5o+Pj4YODAgVJlz549w9ixY1GrVi0IhULY29tjxIgRuHnzZqnjDg0NRcuWLSEUClG/fn0EBgaWWP/Fixfc5/Lzx9WrV7k6s2fPxp49e/D8+fNSx0UoASpVwTW/4OBgqSRIKq6ePXsiISEBz58/x/r16/H777/D399f3WGRQi5fvoyYmBh8/fXXRZb99ddfaNy4MRwcHMrUor558yZcXFzw5MkT/P7773j06BGOHTsGBwcHfP/996XaZmxsLPr06YMuXbogMjIS3333Hb799lucO3fui+ueP38eCQkJ3KNgyiIgf55CDw8PbNmypVRxkXyUAJWkcIeXunXrqjskIgehUAhLS0vY2tpi4MCBcHd3R0hICLf8w4cPGDFiBGxsbKCrq4umTZviwIEDUtv46quvMGPGDMyZMwempqawtLTE4sWLpeo8ffoUnTp1gra2NpycnKT2UeD+/fvo2rUrdHR0UKNGDUyYMIGbVBb4X4tl+fLlqFmzJoyNjbF06VLk5eVx0zLVqlULu3fvLvE1f/r0CSNHjoSenh6srKywfv16fPXVV/juu++4OrJaTQVTFBV4+fIlhg0bBmNjY5iammLAgAF48eIFtzw0NBSurq7Q09ODsbEx2rdvz03eevfuXW66HENDQ7i4uJTYygoKCkL37t1l3nB9586dGDVqFEaNGoWdO3eW+NqLwxiDj48PGjRogLCwMPTp0wf16tWDs7Mz/P398ffff5dqu1u3boW9vT3Wrl0LR0dHTJs2DUOGDMH69eu/uG6NGjVgaWnJPQqPHe7Xrx+CgoJKFRfJRwlQCai3p2w5OTnFPvLy8uSuW3jCz+LqldWDBw9w5coVqZm/s7Ky4OLiglOnTuHBgweYMGECRo8ejevXr0utu2fPHujp6eHatWv45ZdfsHTpUi7JSSQSDB48GAKBANeuXcPWrVuLzN+Xnp4ODw8PmJiY4MaNGzh8+DDOnz+PadOmSdX7999/8ebNG1y6dAnr1q2Dv78/+vbtCxMTE1y7dg2TJk3CxIkT8erVq2Jfp6+vL8LDwxEcHIyQkBCEhYVxk7TKKzc3Fx4eHjAwMEBYWBjCw8Ohr6+Pnj17cu/vwIED0blzZ9y7dw8RERGYMGECN3/eyJEjUatWLdy4cQO3bt2Cn59fiTeHCAsLk3nvyJiYGERERGDYsGEYNmwYwsLCSpwhvTiRkZF4+PAhvv/++yKTvQKQmpC2cePG0NfXL/bRq1cvrm5ERATc3d2ltuXh4YGIiIgvxtS/f39YWFigQ4cOCA4OLrLc1dUVr169kvrRQRRDvUDLiJJf8VasWFHssgYNGsDLy4t7vmbNmmJntq5Tpw58fHy45xs2bOBmpv5caU5dnjx5Evr6+sjLy0N2djb4fD42btzILbexscHs2bO559OnT8e5c+dw6NAhuLq6cuXNmjXj9t+gQQNs3LgRFy5cQPfu3XH+/Hk8fvwY586d4yYaXb58udQX5f79+5GVlYW9e/dCT08PALBx40b069cPq1at4mZiNzU1xa+//go+n49GjRrhl19+QUZGBn788UcAwLx587By5UpcvnwZw4cPL/J6P336hD179mD//v3o1q0bAGD37t1cXPI6ePAgJBIJduzYwSW13bt3w9jYGKGhoWjVqhVEIhH69u3LzRjh6OjIrR8fH48ffvgBDg4O3DErSVxcnMwYd+3ahV69enETrXp4eGD37t1FWuBf8vTpUwDg4imJIrOwJyYmcu9dgZo1ayI1NRWZmZkyZ2zX19fH2rVr0b59e/D5fPz1118YOHAgjh8/jv79+3P1Co5HXFwc7Ozsvhg3KYoSYBlkZ2dT8qvkunTpgi1btiA9PR3r16+Hpqam1HUmsViM5cuX49ChQ3j9+jVycnKQnZ0NXV1dqe00a9ZM6rmVlRXevXsHAIiKioKtra3UF3jheeWioqLQvHlzLvkBQPv27SGRSBAdHc19iTZu3FiqhVKzZk00adKEe66hoYEaNWpw+y7s+fPnyM3NlUreRkZGaNSoUckHqpC7d+/i2bNnMDAwkCrPyspCTEwMevToAR8fH3h4eKB79+5wd3fHsGHDYGVlBSC/Ffrtt9/ijz/+gLu7O4YOHVri1EqZmZlFTn+KxWLs2bMHGzZs4MpGjRqF2bNnY9GiRTJbcsVRZFrUsszCLg8zMzP4+vpyz1u3bo03b95g9erVUgmwIHnK+jFI5EMJsAyEQiFatmyJ27dvU/KTYd68ecUuK/zl9Hkrq7CCFkaBmTNnli2wz+jp6aF+/foA8lsTzZs3x86dO7kJbVevXo0NGzYgICAATZs2hZ6eHr777rsip1wLn77j8XiQSCRKi7Ok/ahi3zwer0hS+LzVk5aWBhcXF+zbt6/Iuubm5gDyW4QzZszA2bNncfDgQSxYsAAhISFo06YNFi9eDC8vL5w6dQpnzpyBv78/goKCMGjQIJnxmJmZ4ePHj1Jl586dw+vXr+Hp6SlVLhaLudY3ABgYGEAkEhXZZkpKCvc327BhQwDA48eP0aJFixKPTePGjUs8zdqxY0ecOXMGAGBpaYm3b99KLX/79i0MDQ1ltv6K4+bmVuS6cXJyMoD/HW+iOEqAZdShQwe0bt0aQqFQ3aFUOJ9fS1NXXUXw+Xz8+OOP8PX1hZeXF3R0dBAeHo4BAwZg1KhRAPKv5z158gROTk5yb9fR0REvX75EQkIC1wL6vEt7QZ3AwECkp6dzrcDw8HDuVKey1K1bF1paWrhx4wZq164NIP80/pMnT9CpUyeunrm5ORISErjnT58+lWpptGzZEgcPHoSFhUWJw3xatGiBFi1aYN68eWjbti3279+PNm3aAMhPOg0bNsSsWbMwYsQI7N69u9gE2KJFCzx69EiqbOfOnRg+fDjmz58vVb5s2TLs3LmTS4CNGjXCrVu34O3tzdURi8W4e/cuvv32WwCAs7MznJycsHbtWnh6ehb5gZaSksJdB1TkFGjbtm1x+vRpqeUhISEKzywfGRnJfXYKPHjwAFpaWmjcuLFC2yL/Q51gFCQSiXD06FGp8UaU/KqOoUOHQkNDA5s2bQKQf20qJCQEV65cQVRUFCZOnFjkF/2XuLu7o2HDhvD29sbdu3cRFhZW5Et75MiR0NbWhre3Nx48eICLFy9i+vTpGD16dJFrSGVhYGAAb29v/PDDD7h48SIePnyIcePGgc/nS7W0u3btio0bN+LOnTu4efMmJk2aJNXSHDlyJMzMzDBgwACEhYUhNjYWoaGhmDFjBl69eoXY2FjMmzcPERERiIuLwz///IOnT5/C0dERmZmZmDZtGkJDQxEXF4fw8HDcuHFD6hphYR4eHrh8+TL3/P379zhx4gS8vb3RpEkTqceYMWNw/PhxroXk6+uLHTt2YPPmzXj69CkiIyMxYcIEfPz4kUuAPB4Pu3fvxpMnT9CxY0ecPn0az58/x71797Bs2TIMGDCA23edOnVQv379Yh82NjZc3UmTJuH58+eYM2cOHj9+jM2bN+PQoUOYNWsWV2fjxo3c9Vggv0PVgQMH8PjxYzx+/BjLly/Hrl27igx8DwsLQ8eOHRVqSRJplAAVUNDh5f79+zh16pS6wyEqoKmpiWnTpuGXX35Beno6FixYgJYtW8LDwwNfffUVLC0tiwye/hI+n49jx44hMzMTrq6u+Pbbb7Fs2TKpOrq6ujh37hySk5PRunVrDBkyBN26dZPqkKMs69atQ9u2bdG3b1+4u7ujffv2cHR0lLrGtnbtWtja2qJjx47w8vLC7Nmzpa576urq4tKlS6hduzYGDx4MR0dHjBs3DllZWTA0NISuri4eP36Mr7/+Gg0bNsSECRMwdepUTJw4ERoaGvjw4QPGjBmDhg0bYtiwYejVqxeWLFlSbMwjR47Ew4cPER0dDQBcZ6HPE0eBbt26QUdHB3/++ScAYMSIEdixYwd27doFFxcX9OzZE4mJibh06ZLUjwtXV1fcvHkT9evXx/jx4+Ho6Ij+/fvj4cOHCAgIKNWxtre3x6lTpxASEoLmzZtj7dq12LFjBzw8PLg6SUlJiImJkVrvp59+gouLC9zc3PD333/j4MGDGDt2rFSdoKAgjB8/vlRxkXw8psjV3yogNTUVRkZGEIlESM7RwFdrQmEg1MT9JR4lrke9PWXLyspCbGws7O3tZY7RIhVfeno6bGxssHbtWu7aZ0X0ww8/IDU1Fb///ru6Q1G7M2fO4Pvvv8e9e/egqVk1r2SV9N3y+fd4We60RS1AOVDyI1XJnTt3cODAAcTExOD27dsYOXIkAEid5quI5s+fjzp16qikc1Flk56ejt27d1fZ5Fde6Oh9ASU/UhWtWbMG0dHREAgEcHFxQVhYGMzMzNQdVomMjY258Y7V3ZAhQ9QdQpVACbAEjDEcOXKEkh+pUlq0aIFbt26pOwxC1I5OgZaAx+OhX79+qFWrFiU/QgipYqgFKINEIuHGAVlYWOCbb74pMhibEEJI5UYtwEJEIhF+//13qRvMUvL7smrWmZgQomLl8Z1CCfAzBR1e3r17hzNnzlBvMzkUDI6m+xESQpSp4DulpFlCyqpCnALdtGkTVq9ejcTERDRv3hy//fab1M16Czt8+DAWLlyIFy9eoEGDBli1ahV69+5dphgK9/b08vJS6Ga61ZWGhgaMjY25my/r6upSi5kQUmqMMWRkZODdu3cwNjaGhoaGyval9gR48OBB+Pr6YuvWrXBzc0NAQAA8PDwQHR0NCwuLIvWvXLmCESNGYMWKFejbty/279+PgQMH4vbt21J3xVeEDmhWh7KwtLQEgGJnICCEEEUZGxtz3y2qovY7wbi5uaF169bcLZ8kEglsbW0xffp0+Pn5Fanv6emJ9PR0nDx5kitr06YNnJ2dsXXr1i/ur/CdYPqs/Qe9hdHQ52VT8isjsVhc4k2CCSFEHlpaWiW2/JR1Jxi1tgBzcnJw69YtqWlz+Hw+3N3di50xOSIiQmquLCD/RrnHjx+XWT87O1vqxtWpqalSy5toJlLyUxINDQ2Vnq4ghBBlUutFrqSkJIjFYpkzJicmJspcp7gZlourv2LFChgZGXEPW1tbqeU3cmvhKbOk5EcIIdWM2q8Bqtq8efOkWoypqalcErQ21sG/s7uCz+PByEi3uE0QQgipgtSaAM3MzKChoSFzxuTiLn4WN8NycfWFQmGx8/UJNPmwM9MrReSEEEIqO7UmwIIb8V64cIGbY00ikeDChQuYNm2azHXatm2LCxcu4LvvvuPKFJlhuaDPT+FrgYQQQiqHgu/vMvfhZGoWFBTEhEIhCwwMZI8ePWITJkxgxsbGLDExkTHG2OjRo5mfnx9XPzw8nGlqarI1a9awqKgo5u/vz7S0tNj9+/fl2t/Lly8ZAHrQgx70oEclf7x8+bJM+Uft1wA9PT3x/v17LFq0CImJiXB2dsbZs2e5ji7x8fFSA9LbtWuH/fv3Y8GCBfjxxx/RoEEDHD9+XO4xgNbW1nj58iUMDAzA4/G4a4IvX74sU3faqoqOz5fRMSoZHZ8vo2NUssLHhzGGT58+wdraukzbVfs4QHVT1niSqoqOz5fRMSoZHZ8vo2NUMlUdH7rXFyGEkGqJEiAhhJBqqdonQKFQCH9//2KHSlR3dHy+jI5Ryej4fBkdo5Kp6vhU+2uAhBBCqqdq3wIkhBBSPVECJIQQUi1RAiSEEFItUQIkhBBSLVWLBLhp0ybY2dlBW1sbbm5uuH79eon1Dx8+DAcHB2hra6Np06Y4ffp0OUWqHoocn+3bt6Njx44wMTGBiYkJ3N3dv3g8qwJFP0MFgoKCwOPxuHvdVlWKHp+UlBRMnToVVlZWEAqFaNiwIf2dFRIQEIBGjRpBR0cHtra2mDVrFrKyssop2vJ16dIl9OvXD9bW1uDxeMXO7/q50NBQtGzZEkKhEPXr10dgYKDiOy7TjdQqgaCgICYQCNiuXbvYw4cP2fjx45mxsTF7+/atzPrh4eFMQ0OD/fLLL+zRo0dswYIFCt1rtLJR9Ph4eXmxTZs2sTt37rCoqCjm4+PDjIyM2KtXr8o58vKj6DEqEBsby2xsbFjHjh3ZgAEDyidYNVD0+GRnZ7NWrVqx3r17s8uXL7PY2FgWGhrKIiMjyzny8qPoMdq3bx8TCoVs3759LDY2lp07d45ZWVmxWbNmlXPk5eP06dNs/vz57OjRowwAO3bsWIn1nz9/znR1dZmvry979OgR++2335iGhgY7e/asQvut8gnQ1dWVTZ06lXsuFouZtbU1W7Fihcz6w4YNY3369JEqc3NzYxMnTlRpnOqi6PEpLC8vjxkYGLA9e/aoKkS1K80xysvLY+3atWM7duxg3t7eVToBKnp8tmzZwurWrctycnLKK0S1U/QYTZ06lXXt2lWqzNfXl7Vv316lcVYE8iTAOXPmsMaNG0uVeXp6Mg8PD4X2VaVPgebk5ODWrVtwd3fnyvh8Ptzd3RERESFznYiICKn6AODh4VFs/cqsNMensIyMDOTm5sLU1FRVYapVaY/R0qVLYWFhgXHjxpVHmGpTmuMTHByMtm3bYurUqahZsyaaNGmC5cuXQywWl1fY5ao0x6hdu3a4desWd5r0+fPnOH36NHr37l0uMVd0yvqeVvtsEKqUlJQEsVjMzSxRoGbNmnj8+LHMdRITE2XWT0xMVFmc6lKa41PY3LlzYW1tXeTDWFWU5hhdvnwZO3fuRGRkZDlEqF6lOT7Pnz/Hv//+i5EjR+L06dN49uwZpkyZgtzcXPj7+5dH2OWqNMfIy8sLSUlJ6NChAxhjyMvLw6RJk/Djjz+WR8gVXnHf06mpqcjMzISOjo5c26nSLUCiWitXrkRQUBCOHTsGbW1tdYdTIXz69AmjR4/G9u3bYWZmpu5wKiSJRAILCwts27YNLi4u8PT0xPz587F161Z1h1ZhhIaGYvny5di8eTNu376No0eP4tSpU/jpp5/UHVqVUqVbgGZmZtDQ0MDbt2+lyt++fQtLS0uZ61haWipUvzIrzfEpsGbNGqxcuRLnz59Hs2bNVBmmWil6jGJiYvDixQv069ePK5NIJAAATU1NREdHo169eqoNuhyV5jNkZWUFLS0taGhocGWOjo5ITExETk4OBAKBSmMub6U5RgsXLsTo0aPx7bffAgCaNm2K9PR0TJgwAfPnz5eaI7U6Ku572tDQUO7WH1DFW4ACgQAuLi64cOECVyaRSHDhwgW0bdtW5jpt27aVqg8AISEhxdavzEpzfADgl19+wU8//YSzZ8+iVatW5RGq2ih6jBwcHHD//n1ERkZyj/79+6NLly6IjIyEra1teYavcqX5DLVv3x7Pnj3jfhgAwJMnT2BlZVXlkh9QumOUkZFRJMkV/GBgdPtm5X1PK9Y/p/IJCgpiQqGQBQYGskePHrEJEyYwY2NjlpiYyBhjbPTo0czPz4+rHx4ezjQ1NdmaNWtYVFQU8/f3r/LDIBQ5PitXrmQCgYAdOXKEJSQkcI9Pnz6p6yWonKLHqLCq3gtU0eMTHx/PDAwM2LRp01h0dDQ7efIks7CwYD///LO6XoLKKXqM/P39mYGBATtw4AB7/vw5++eff1i9evXYsGHD1PUSVOrTp0/szp077M6dOwwAW7duHbtz5w6Li4tjjDHm5+fHRo8ezdUvGAbxww8/sKioKLZp0yYaBlGc3377jdWuXZsJBALm6urKrl69yi3r3Lkz8/b2lqp/6NAh1rBhQyYQCFjjxo3ZqVOnyjni8qXI8alTpw4DUOTh7+9f/oGXI0U/Q5+r6gmQMcWPz5UrV5ibmxsTCoWsbt26bNmyZSwvL6+coy5fihyj3NxctnjxYlavXj2mra3NbG1t2ZQpU9jHjx/LP/BycPHiRZnfKwXHxNvbm3Xu3LnIOs7OzkwgELC6deuy3bt3K7xfmg6JEEJItVSlrwESQgghxaEESAghpFqiBEgIIaRaogRICCGkWqIESAghpFqiBEgIIaRaogRICCGkWqIESAghpFqiBEiKFRgYCGNjY3WHUWo8Hg/Hjx8vsY6Pjw8GDhxYLvFUNAsXLsSECRPKZV+hoaHg8XhISUkpsZ6dnR0CAgJUGoui+1DW34E8n0dFPXr0CLVq1UJ6erpSt1tdUAKs4nx8fMDj8Yo8nj17pu7QEBgYyMXD5/NRq1YtjB07Fu/evVPK9hMSEtCrVy8AwIsXL8Dj8YrM0bdhwwYEBgYqZX/FWbx4Mfc6NTQ0YGtriwkTJiA5OVmh7SgzWScmJmLDhg2YP3++1PYL4hQIBKhfvz6WLl2KvLy8Mu+vXbt2SEhIgJGREYDik8qNGzfKLSlXBsuWLUO7du2gq6sr83g5OTmhTZs2WLduXfkHVwVQAqwGevbsiYSEBKmHvb29usMCABgaGiIhIQGvXr3C9u3bcebMGYwePVop27a0tIRQKCyxjpGRUbm0chs3boyEhATEx8dj9+7dOHv2LCZPnqzy/RZnx44daNeuHerUqSNVXvBZefr0Kb7//nssXrwYq1evLvP+BAIBLC0twePxSqxnbm4OXV3dMu+vqsjJycHQoUNL/KyMHTsWW7ZsUcoPleqGEmA1IBQKYWlpKfXQ0NDAunXr0LRpU+jp6cHW1hZTpkxBWlpasdu5e/cuunTpAgMDAxgaGsLFxQU3b97kll++fBkdO3aEjo4ObG1tMWPGjC+emuHxeLC0tIS1tTV69eqFGTNm4Pz588jMzIREIsHSpUtRq1YtCIVCODs74+zZs9y6OTk5mDZtGqysrKCtrY06depgxYoVUtsuOOVUkPBbtGgBHo+Hr776CoB0q2rbtm2wtraWmqYHAAYMGIBvvvmGe/7333+jZcuW0NbWRt26dbFkyZIvfvloamrC0tISNjY2cHd3x9ChQxESEsItF4vFGDduHOzt7aGjo4NGjRphw4YN3PLFixdjz549+Pvvv7lWWmhoKADg5cuXGDZsGIyNjWFqaooBAwbgxYsXJcYTFBQkNWdhgYLPSp06dTB58mS4u7sjODgYAPDx40eMGTMGJiYm0NXVRa9evfD06VNu3bi4OPTr1w8mJibQ09ND48aNcfr0aQDSp0BDQ0MxduxYiEQi7rUsXrwYgPTpSS8vL3h6ekrFl5ubCzMzM+zduxdA/rRCK1as4I5b8+bNceTIkRJfe2Hy/h0cP34cDRo0gLa2Njw8PPDy5Uup5aX5XHzJkiVLMGvWLDRt2rTYOt27d0dycjL++++/Mu2rOqIEWI3x+Xz8+uuvePjwIfbs2YN///0Xc+bMKbb+yJEjUatWLdy4cQO3bt2Cn58ftLS0AORPBNuzZ098/fXXuHfvHg4ePIjLly9j2rRpCsWko6MDiUSCvLw8bNiwAWvXrsWaNWtw7949eHh4oH///tyX7q+//org4GAcOnQI0dHR2LdvH+zs7GRu9/r16wCA8+fPIyEhAUePHi1SZ+jQofjw4QMuXrzIlSUnJ+Ps2bMYOXIkACAsLAxjxozBzJkz8ejRI/z+++8IDAzEsmXL5H6NL168wLlz56TmvpNIJKhVqxYOHz6MR48eYdGiRfjxxx9x6NAhAMDs2bMxbNgwqdZ8u3btkJubCw8PDxgYGCAsLAzh4eHQ19dHz549kZOTI3P/ycnJePTokVxzOero6HDb8fHxwc2bNxEcHIyIiAgwxtC7d2/k5uYCAKZOnYrs7GxcunQJ9+/fx6pVq6Cvr19km+3atUNAQADX+k9ISMDs2bOL1Bs5ciROnDghlYzOnTuHjIwMDBo0CACwYsUK7N27F1u3bsXDhw8xa9YsjBo1SqFkIM/fQUZGBpYtW4a9e/ciPDwcKSkpGD58OLe8NJ+Lr776Cj4+PnLHWRyBQABnZ2eEhYWVeVvVThlnsSAVnLe3N9PQ0GB6enrcY8iQITLrHj58mNWoUYN7vnv3bmZkZMQ9NzAwYIGBgTLXHTduHJswYYJUWVhYGOPz+SwzM1PmOoW3/+TJE9awYUPWqlUrxhhj1tbWbNmyZVLrtG7dmk2ZMoUxxtj06dNZ165dmUQikbl9AOzYsWOMMcZiY2MZAHbnzh2pOoWnKhowYAD75ptvuOe///47s7a2ZmKxmDHGWLdu3djy5cultvHHH38wKysrmTEwlj+3G5/PZ3p6ekxbW5ub6mXdunXFrsMYY1OnTmVff/11sbEW7LtRo0ZSxyA7O5vp6Oiwc+fOydxuwZxr8fHxUuWfb18ikbCQkBAmFArZ7Nmz2ZMnTxgAFh4eztVPSkpiOjo67NChQ4wxxpo2bcoWL14sc58F090UTOdT+L0vUKdOHbZ+/XrGWP6UQGZmZmzv3r3c8hEjRjBPT0/GGGNZWVlMV1eXXblyRWob48aNYyNGjJAZR+F9yCLr7wCA1PRFUVFRDAC7du0aY0y+z8Xnn0fGvjyP5OeKO14FBg0axHx8fOTaFvkfTXUlXlJ+unTpgi1btnDP9fT0AOS3hlasWIHHjx8jNTUVeXl5yMrKQkZGhszrML6+vvj222/xxx9/cKfx6tWrByD/9Oi9e/ewb98+rj5jDBKJBLGxsXB0dJQZm0gkgr6+PiQSCbKystChQwfs2LEDqampePPmDdq3by9Vv3379rh79y6A/BZJ9+7d0ahRI/Ts2RN9+/ZFjx49ynSsRo4cifHjx2Pz5s0QCoXYt28fhg8fzs3OfffuXYSHh0v9sheLxSUeNwBo1KgRgoODkZWVhT///BORkZGYPn26VJ1NmzZh165diI+PR2ZmJnJycuDs7FxivHfv3sWzZ89gYGAgVZ6VlYWYmBiZ62RmZgIAtLW1iyw7efIk9PX1kZubC4lEAi8vLyxevBgXLlyApqYm3NzcuLo1atRAo0aNEBUVBQCYMWMGJk+ejH/++Qfu7u74+uuv0axZsxLjL4mmpiaGDRuGffv2YfTo0UhPT8fff/+NoKAgAMCzZ8+QkZGB7t27S62Xk5ODFi1ayL0fef4ONDU10bp1a24dBwcHGBsbIyoqCq6urqX6XBScxlUGHR0dZGRkKG171QUlwGpAT08P9evXlyp78eIF+vbti8mTJ2PZsmUwNTXF5cuXMW7cOOTk5Mj8g128eDG8vLxw6tQpnDlzBv7+/ggKCsKgQYOQlpaGiRMnYsaMGUXWq127drGxGRgY4Pbt2+Dz+bCysoKOjg4AIDU19Yuvq2XLloiNjcWZM2dw/vx5DBs2DO7u7gpfA/pcv379wBjDqVOn0Lp1a4SFhWH9+vXc8rS0NCxZsgSDBw8usq6shFKgoFclAKxcuRJ9+vTBkiVL8NNPPwHIvyY3e/ZsrF27Fm3btoWBgQFWr16Na9eulRhvWloaXFxcpH54FDA3N5e5jpmZGYD8a3qF6xT8WBIIBLC2toampvxfEd9++y08PDxw6tQp/PPPP1ixYgXWrl1bJNErYuTIkejcuTPevXuHkJAQ6OjooGfPngDAnRo9deoUbGxspNb7UuenAqX5O5CltJ8LZUlOTuZ+jBL5UQKspm7dugWJRIK1a9dyrZuC600ladiwIRo2bIhZs2ZhxIgR2L17NwYNGoSWLVvi0aNHRRLtl/D5fJnrGBoawtraGuHh4ejcuTNXHh4eDldXV6l6np6e8PT0xJAhQ9CzZ08kJyfD1NRUansF19vEYnGJ8Whra2Pw4MHYt28fnj17hkaNGqFly5bc8pYtWyI6Olrh11nYggUL0LVrV0yePJl7ne3atcOUKVO4OoVbcAKBoEj8LVu2xMGDB2FhYQFDQ0O59l2vXj0YGhri0aNHaNiwodQyWT+WAMDR0RF5eXm4du0a2rVrBwD48OEDoqOj4eTkxNWztbXFpEmTMGnSJMybNw/bt2+XmQBlvRZZ2rVrB1tbWxw8eBBnzpzB0KFDuevOTk5OEAqFiI+Pl/qMKELev4O8vDzcvHmT++xFR0cjJSWFO7OhrM9FaT148ABDhgxRy74rM+oEU03Vr18fubm5+O233/D8+XP88ccf2Lp1a7H1MzMzMW3aNISGhiIuLg7h4eG4ceMG9wUwd+5cXLlyBdOmTUNkZCSePn2Kv//+W+FOMJ/74YcfsGrVKhw8eBDR0dHw8/NDZGQkZs6cCSC/996BAwfw+PFjPHnyBIcPH4alpaXMYQ0WFhbQ0dHB2bNn8fbtW4hEomL3O3LkSJw6dQq7du3iOr8UWLRoEfbu3YslS5bg4cOHiIqKQlBQEBYsWKDQa2vbti2aNWuG5cuXAwAaNGiAmzdv4ty5c3jy5AkWLlyIGzduSK1jZ2eHe/fuITo6GklJScjNzcXIkSNhZmaGAQMGICwsDLGxsQgNDcWMGTPw6tUrmfvm8/lwd3fH5cuX5Y63QYMGGDBgAMaPH4/Lly/j7t27GDVqFGxsbDBgwAAAwHfffYdz584hNjYWt2/fxsWLF4s99W1nZ4e0tDRcuHABSUlJJZ6+8/LywtatWxESEiL1fhgYGGD27NmYNWsW9uzZg5iYGNy+fRu//fYb9uzZI9frkvfvQEtLC9OnT8e1a9dw69Yt+Pj4oE2bNlxCLM3nYsyYMZg3b16J8cXHxyMyMhLx8fEQi8WIjIxEZGSkVMegFy9e4PXr13B3d5frNZPPqPsiJFEtWR0nCqxbt45ZWVkxHR0d5uHhwfbu3VtsR4Xs7Gw2fPhwZmtrywQCAbO2tmbTpk2T6uBy/fp11r17d6avr8/09PRYs2bNinRi+dyXLuyLxWK2ePFiZmNjw7S0tFjz5s3ZmTNnuOXbtm1jzs7OTE9PjxkaGrJu3bqx27dvc8tRqNPB9u3bma2tLePz+axz587FHh+xWMysrKwYABYTE1MkrrNnz7J27doxHR0dZmhoyFxdXdm2bduKfR3+/v6sefPmRcoPHDjAhEIhi4+PZ1lZWczHx4cZGRkxY2NjNnnyZObn5ye13rt377jjC4BdvHiRMcZYQkICGzNmDDMzM2NCoZDVrVuXjR8/nolEomJjOn36NLOxseE69xR3LD6XnJzMRo8ezYyMjLjPzJMnT7jl06ZNY/Xq1WNCoZCZm5uz0aNHs6SkJMZY0U4wjDE2adIkVqNGDQaA+fv7M8Zkd1B59OgRA8Dq1KlTpMOTRCJhAQEBrFGjRkxLS4uZm5szDw8P9t9//xX7OgrvQ96/g7/++ovVrVuXCYVC5u7uzuLi4qS2+6XPReHPY+fOnZm3t3excTKW/57g/ztNff4oeO8ZY2z58uXMw8OjxO0Q2XiMMaaOxEsIUR/GGNzc3LhT2aRyysnJQYMGDbB///4iHcbIl9EpUEKqIR6Ph23bttHdQyq5+Ph4/Pjjj5T8SolagIQQQqolagESQgipligBEkIIqZYoARJCCKmWKAESQgipligBEkIIqZYoARJCCKmWKAESQgipligBEkIIqZYoARJCCKmW/g96aGfLEMjDzAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "B_1zhmZ48puy"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}