import streamlit as st
import plotly.express as px

def app():
    st.image('./Streamlit_UI/arabic.jpg', use_column_width=False)

    st.subheader("💡 Abstract:")

    inspiration = '''
    For the Hogeschool Utrecht Machine Learning course of the summer of 2023 I have developed a deep learning pipeline to classify spoken Arabic digits.   
    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What the Project Does?")

    what_it_does = '''
    Simply submit a wav file of a spoken Arabic digit, and the machine learning model will evaluate it and provide a response in a fraction of a second.'''

    st.markdown(what_it_does, unsafe_allow_html=True)

    stats, buff, graph = st.columns([2, 0.5, 2])

    stats.subheader("🧠 ML Process")

    stats.markdown("*Data Collected by the Laboratory of Automatic and Signals, University of Badji-Mokhtar Annaba, Algeria. You can find it [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00195/).*")

    stats.markdown('''This dataset contains time series of mel-frequency cepstrum coefficients (MFCCs) corresponding to spoken Arabic digits. 
Includes data from 44 males and 44 females native Arabic speakers.

Dataset from 8800 (10 digits x 10 repetitions x 88 speakers) time series of 13 Frequency Cepstral Coefficients (MFCCs) had taken from 44 males and 44 females Arabic native speakers between the ages 18 and 40 to represent ten spoken Arabic digit.

Each line on the data base represents 13 MFCCs coefficients in the increasing order separated by spaces. This corresponds to one analysis frame. The 13 Mel Frequency Cepstral Coefficients (MFCCs) are computed with the following conditions; Sampling rate: 11025 Hz, 16 bits Window applied: hamming Filter pre-emphasized: 1-0.97Z^(-1)
    ''', unsafe_allow_html=True) 

    st.subheader("⚙️ Model Architecture")

    st.markdown('''### Pipeline''')
    st.text('''
AttentionGRUAarabic(
  (rnn): GRU(((13, 120, num_layers=4, batch_first=True, dropout=0.02319741641270231)
  (attention): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=120, out_features=120, bias=True)
  )
  (linear): Linear(in_features=120, out_features=20, bias=True)
)''')

    st.image('./Streamlit_UI/GRUAttentionArabicModel.png', use_column_width=False)

    ml_process = f'''Various architectures where on the short list for this model. After trying a LSTM with limited success and a GRU with
    increasing results, I settled on a combination of a GRU layer followed by a MultiheadAttention layer:

- I designed a Sequential Model having a GRU Layer, a MultiheadAttention Layer and a Linear Layer.
- The GRU Layer starts with an input_size of 13 filters, one for each time series of 13 Frequency Cepstral Coefficients (MFCCs).
    - A hidden_size of 120 seemed to work best. This value needed to be compatible - divisible by  with the number of heads of the Attention Layer.
    - The dropout was very low, at: 0.023197
    - 4 layers were used.
- The MultiheadAttention Layer receives the output of the GRU Layer.
    - This layer also has a hidden_size of 120 for compatibility in the pipeline with the GRU.
    - The number of heads was 30.
    - The dropout was very low, at: 0.023197
- The Linear Layer receives the output of the MultiheadAttention Layer and flattens it to the desired number of categories.
    Hence, the output_size was 20: 10 digits spoken with a male identified voice, and 10 digits spoken with a female identified voice.
'''
    st.write(ml_process)

    st.markdown('''### Hypertuning''')

    ml_hypertunig = f'''Hypertuning was done on the following hyperparameters:
- hidden_size and nheads, values between 4 and 128 were tested. these were hyptuned together as these values needed to be aligned.
    - Result: hidden_size: 30
    - Result: num_heads: 120
- dropout_gru: between 0.0 and 0.3 was tested for the dropout of the GRU Layer
    - Result: 0.123935
- dropout_attention: between 0.0 and 0.3 was tested for the dropout of the Attention Layer
    - Result: 0.202798
- num_layers: between 2 and 5 layers were tested for the GRU layers
    - Result: 4
- use_mean: using mean of last_step was attempted for the forward function.
    - Result: False (last_step used)

These parameters were not hypertuned due to low added value expected in this use case:
- CrossEntropyLoss as the loss function, as this is the current standard for multi-class predictions. Further optimization could be tried with a loss function like Categorical Cross-Entropy.  
- Adam was used as an opitimizer. Possible alternatives that were not tried are: Stochastic Gradient Descent (SGD), Momentum, Nesterov Accelerated Gradient (NAG), RMSProp, AdaGrad, AdaDelta, Adam, Nadam, AdaMax, AMSGrad, L-BFGS.
- ReduceLROnPlateau was used to tune the learning_rate
- Epochs. It was clear that after about 50 epochs training was done. Since ReduceLROnPlateau was implemented, no large change was expected after this amount. Also the risk overfitting with a larger number of epochs increases.
'''
    st.write(ml_hypertunig)

    results = f'''
    - The model with least Validation Loss was saved during the training and reloaded before obtaining the final results.
    - The model was able to classify all of the samples correctly.
    '''
    loss, buff, acc = st.columns([2, 0.4, 2])

    # loss.image('./Streamlit_UI/loss.png', use_column_width=True)
    # acc.image('./Streamlit_UI/accuracy.png', use_column_width=True)

    st.subheader("📈 Results")
    st.markdown(results, unsafe_allow_html=True)

    st.write("Classification Report:")
    
    cfr = '''
 Report Title     precision    recall  f1-score   support

        Real       1.00      1.00      1.00        59
        Fake       1.00      1.00      1.00        70

    accuracy                           1.00       129
   macro avg       1.00      1.00      1.00       129
weighted avg       1.00      1.00      1.00       129
'''
    st.code(cfr)

    st.write(" ")

    st.write("*Try it out now by clicking on Classify Image button on the Sidebar*")
