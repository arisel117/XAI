# 설명 가능한 인공지능 (XAI, Explainable Artificial Intelligence)


## 주요 목표
- AI의 의사 결정에 대해서 언제, 어떻게, 왜 그런 결정을 하였는지 설명이 가능해야함
- 이를 통해 신뢰성, 책임성, 공정성, 규제 준수를 할 수 있음


## 화이트 박스 모델
- 내재적으로 설명이 가능한 모델로, 주로 XGBoost(Decision Tree), 선형 회귀, 로지스틱 회귀 등이 있음
- 사용자가 모델이 어떻게 작동하는지를 이해 할 수 있고, 주어진 조건에 의해서 동작하므로 모델이 설명 가능함


## 블랙 박스 모델
- 내재적으로 설명하기 어려운 모델로, 딥러닝이나 복잡한 앙상블 모델이 이에 해당함
- 예측 성능은 우수하지만 내부의 작동 방식이 복잡하여 설명이 어려움
- Transformer의 경우 Encoder / Decoder Block을 Residual 연산을 통해 연결하면서, Attention Map을 분석하여 영향도를 계산하기도 함

### LIME (Local Interpretable Model-agnostic Explanations)
- 개별 예측을 설명하기 위해 원래 모델과 유사하게 작동하는 단순한 모델을 사용하여 특정 데이터 포인트 주변의 예측 결과로 통계적으로 해석을 제공함

### SHAP (SHapley Additive exPlanations)
- 게임 이론에 기반한 접근 방법으로, 각 특징점이 모델의 예측에 얼마나 기여했는지를 평가함


## XAI with Code
- PyTorch Captum XAI
  - [PyTorch Doc](https://tutorials.pytorch.kr/recipes/recipes/Captum_Recipe.html)
  - [Captum Doc](https://captum.ai/docs/introduction)
  - [주요 사용 가능한 알고리즘](https://captum.ai/docs/algorithms_comparison_matrix#attribution-algorithm-comparison-matrix)은 링크게 자세하게 설명되어 있으며, 그 중에서도 주로 사용하는 알고리즘은 다음과 같음
    - Integrated Gradients / Input * Gradient
    - LayerGradCam / GuidedGradCam
    - Occlusion
    - GradientSHAP / kernelSHAP / DeepLiftSHAP
    - 이 밖에도 다양한 알고리즘을 제공함

- 특히, NLP, Computer Vision 분야에서 모델의 예측 결과에 대한 해석 기능을 시각화 기능을 포함하여 알기 쉽게 제공함
  - [LIME](https://github.com/marcotcr/lime)
  - [SHAP](https://shap.readthedocs.io/en/latest/index.html#)


- Tensorflow Integrated Gradients
  ```python
  import numpy as np
  import tensorflow as tf
  
  # Load trained model
  model_path = "./model.hdf5"
  model = tf.keras.models.load_model(model_path, compile=False)
  
  # Load infer of test data
  test_x = np.random.random((1, 100, 100))    # batch, features, features
  test_y = np.random.random((1, 10))          # batch, categories
  
  # Predict data
  pred_score = model.predict(test_x, verbose=0)
  pred_y = tf.math.argmax(pred_score, axis=-1)
  real_y = np.argmax(test_y, axis=-1)
  
  # Get gradients
  with tf.GradientTape() as tape:
      tape.watch(test_x)
      predictions = model(test_x)    # if model have embedding layer, need split model
      loss = tf.gather(predictions, pred_y, axis=1, batch_dims=1)
  gradients = tape.gradient(loss, predictions)
  ```
