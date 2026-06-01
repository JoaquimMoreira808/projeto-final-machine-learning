import streamlit as st
import pandas as pd
import joblib
import numpy as np

# configuracoes da pagina
st.set_page_config(
    page_title="Previsao de Preco de Carros",
    layout="wide"
)

# carregamento do modelo salvo
@st.cache_resource
def load_model():
    return joblib.load('model/modelo_final.pkl')

try:
    pipeline = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Erro ao carregar o modelo: {e}")

st.title('Previsao de Preco de Carros')
st.write("""
Aplicacao que utiliza um modelo de regressao linear treinado com o dataset
CarPrice Assignment (1985) para estimar o preco de um veiculo com base nas
suas caracteristicas tecnicas.
""")

st.header("Caracteristicas do veiculo")

# layout em 3 colunas
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Informacoes basicas")
    brand = st.selectbox('Marca', ['alfa-romeo', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'buick', 'mercury', 'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'])
    fueltype = st.selectbox('Tipo de combustivel', ['gas', 'diesel'])
    aspiration = st.selectbox('Aspiracao', ['std', 'turbo'])
    doornumber = st.selectbox('Numero de portas', ['two', 'four'])
    carbody = st.selectbox('Tipo de carroceria', ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'])
    drivewheel = st.selectbox('Tracao', ['rwd', 'fwd', '4wd'])
    enginelocation = st.selectbox('Local do motor', ['front', 'rear'])
    symboling = st.slider('Indice de risco (symboling)', -2, 3, 0)

with col2:
    st.subheader("Dimensoes e peso")
    wheelbase = st.number_input('Distancia entre eixos (wheelbase)', min_value=80.0, max_value=130.0, value=98.0, step=0.1)
    carlength = st.number_input('Comprimento (carlength)', min_value=140.0, max_value=210.0, value=174.0, step=0.1)
    carwidth = st.number_input('Largura (carwidth)', min_value=60.0, max_value=75.0, value=65.9, step=0.1)
    carheight = st.number_input('Altura (carheight)', min_value=45.0, max_value=65.0, value=53.7, step=0.1)
    curbweight = st.number_input('Peso (curbweight)', min_value=1400, max_value=4200, value=2500, step=10)

with col3:
    st.subheader("Motor e desempenho")
    enginetype = st.selectbox('Tipo de motor', ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv'])
    cylindernumber = st.selectbox('Numero de cilindros', ['four', 'six', 'five', 'three', 'twelve', 'two', 'eight'])
    enginesize = st.number_input('Tamanho do motor (enginesize)', min_value=60, max_value=330, value=120, step=1)
    fuelsystem = st.selectbox('Sistema de combustivel', ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi'])
    boreratio = st.number_input('Bore ratio', min_value=2.5, max_value=4.0, value=3.3, step=0.01)
    stroke = st.number_input('Stroke', min_value=2.0, max_value=4.5, value=3.2, step=0.01)
    compressionratio = st.number_input('Taxa de compressao', min_value=7.0, max_value=23.0, value=10.0, step=0.1)
    horsepower = st.number_input('Cavalos de potencia (horsepower)', min_value=40, max_value=300, value=100, step=1)
    peakrpm = st.number_input('RPM de pico', min_value=4000, max_value=6600, value=5100, step=100)
    citympg = st.number_input('Consumo na cidade (citympg)', min_value=10, max_value=50, value=25, step=1)
    highwaympg = st.number_input('Consumo na estrada (highwaympg)', min_value=15, max_value=55, value=30, step=1)

# monta o dataframe com os dados informados pelo usuario
input_dict = {
    'symboling': [symboling],
    'fueltype': [fueltype],
    'aspiration': [aspiration],
    'doornumber': [doornumber],
    'carbody': [carbody],
    'drivewheel': [drivewheel],
    'enginelocation': [enginelocation],
    'wheelbase': [wheelbase],
    'carlength': [carlength],
    'carwidth': [carwidth],
    'carheight': [carheight],
    'curbweight': [curbweight],
    'enginetype': [enginetype],
    'cylindernumber': [cylindernumber],
    'enginesize': [enginesize],
    'fuelsystem': [fuelsystem],
    'boreratio': [boreratio],
    'stroke': [stroke],
    'compressionratio': [compressionratio],
    'horsepower': [horsepower],
    'peakrpm': [peakrpm],
    'citympg': [citympg],
    'highwaympg': [highwaympg],
    'brand': [brand]
}

input_df = pd.DataFrame(input_dict)

st.markdown("---")

# botao de predicao
if st.button('Prever preco', type='primary'):
    if model_loaded:
        with st.spinner('Calculando...'):
            try:
                prediction = pipeline.predict(input_df)[0]

                st.success('Previsao realizada.')

                # exibe o preco estimado
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: #2e4053; margin: 0;'>Preco Estimado</h2>
                    <h1 style='color: #117a65; margin: 10px 0;'>US$ {prediction:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)

                # interpretacao do resultado por faixa de preco
                st.subheader("Interpretacao")
                if prediction < 7500:
                    faixa = "Economico"
                    descricao = "Faixa de preco mais acessivel. Carros compactos, motores menores e menor potencia."
                elif prediction < 15000:
                    faixa = "Intermediario"
                    descricao = "Faixa intermediaria, tipica de sedas e hatchbacks de medio porte."
                elif prediction < 25000:
                    faixa = "Premium"
                    descricao = "Faixa premium, com motores maiores, mais potencia e acabamento superior."
                else:
                    faixa = "Luxo"
                    descricao = "Faixa de luxo, caracteristica de marcas como BMW, Jaguar e Porsche."

                st.info(f"Faixa de preco: {faixa} — {descricao}")

                st.caption("Modelo treinado com dados de 1985. R2 = 0.89 | MAE = US$ 986.55. Os valores nao refletem precos atuais de mercado.")

            except Exception as e:
                st.error(f"Erro durante a previsao: {e}")
    else:
        st.error("Modelo nao carregado. Verifique se o arquivo modelo_final.pkl existe na pasta model/.")
