import re
import warnings
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI as ChatGemini
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)



def chat_with_llm(query, dataframe):
    # Update system prompt to include dataset path information
    system_prompt = f"""### PERSONA ###
Você é um Analista de Dados Sênior, especialista em Python, Pandas e Matplotlib. Seu objetivo é traduzir dados brutos contidos em um DataFrame `df` em respostas claras, objetivas e acionáveis.

### DIRETRIZES GERAIS ###
1.  **Fonte da Verdade:** Baseie 100% de suas análises e conclusões estritamente nos dados do DataFrame. Não faça suposições externas.
2.  **Linguagem:** Responda sempre em português do Brasil.
3.  **Incerteza:** Se uma informação não puder ser obtida a partir dos dados, informe de maneira clara: "Com base nos dados disponíveis, não é possível fornecer essa informação."

### FORMATO DA RESPOSTA ###
1.  **NÃO EXIBA O CÓDIGO:** Sua principal regra é NUNCA exibir o código Python gerado. Apresente apenas os resultados finais, como texto, tabelas e gráficos.
2.  **Clareza e Objetividade:** Seja direto e use uma linguagem de negócios. Evite jargões excessivamente técnicos na sua explicação.
3.  **Uso de Tabelas:** Sempre que apresentar dados com mais de uma coluna ou listas de informações, organize-os em tabelas formatadas em Markdown para máxima legibilidade.
4.  **Precisão Numérica:** Arredonde todos os valores numéricos (flutuantes) para 2 casas decimais.
5.  **Visualizações Estratégicas:** Utilize gráficos (Matplotlib/Seaborn) quando eles forem a melhor forma de ilustrar um ponto, especialmente para:
    - **Distribuições:** Histogramas ou KDE.
    - **Comparações:** Gráficos de barras ou boxplots.
    - **Relações:** Gráficos de dispersão (scatter plots).
6.  **Conclusão Explícita:** Após apresentar os dados e/ou gráficos, inclua sempre um parágrafo conciso de conclusão, explicando o que os resultados significam em termos práticos.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    with st.spinner('Processando as respostas do Gemini...'):
        llm = ChatGemini(model=st.session_state.model_name,
                         google_api_key=st.session_state.gemini_api_key,
                         temperature=0,
                         convert_system_message_to_human=True)
        
        agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        #prompt=messages,
        #verbose=False,
        allow_dangerous_code=True,  # Necessário para execução de código
        agent_executor_kwargs={"handle_parsing_errors": True})
        
        return agent


def main():
    """Main Streamlit application."""
    st.title("📊 Agente de Análise Exploratória de Dados")
    st.write("Faça perguntas sobre um arquivo CSV!")

    # Initialize session state variables
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("Chaves de API e configuração do Modelo")
        st.session_state.gemini_api_key = st.sidebar.text_input("Chave de API do Gemini", type="password")
   
        # Add model selection dropdown
        model_options = {
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.0 Flash": "gemini-2.0-flash",
            "Gemini 1.5 Flash": "gemini-1.5-flash",
        }
        st.session_state.model_name = st.selectbox(
            "Selecione o modelo",
            options=list(model_options.keys()),
            index=0  # Default to first option
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    uploaded_file = st.file_uploader("Faça o upload de um arquivo csv", type="csv")
    
    if uploaded_file is not None:
        # Display dataset with toggle
        df = pd.read_csv(uploaded_file)
        st.write("Dataset:")
        show_full = st.checkbox("Dataset Carregado com sucesso!")
        if show_full:
            st.dataframe(df)
        else:
            st.write("Preview (primeiras 5 linhas):")
            st.dataframe(df.head())
        # Query input
        query = st.text_area("O que você gostaria de saber sobre seus dados?",
                            "Traga os valores máximos de cada coluna do banco de dados.")
        
        if st.button("Análise"):
            if not st.session_state.gemini_api_key:
                st.error("Por favor digite suas chaves de API no campo ao lado.")
            else:
                plt.clf()
                # Pass dataset_path to chat_with_llm
                agent = chat_with_llm(query, df)
                result = agent.invoke({"input": query})
                response = result.get("output", "")

                current_fig = plt.gcf()
                if current_fig.get_axes():
                    st.pyplot(current_fig)
                    st.markdown(response)
                    # Salva com plot na mensagem
                    #ai_msg = AIMessage(content=response, additional_kwargs={"plot": current_fig})
                else:
                    st.markdown(response)
                    #ai_msg = AIMessage(content=response)
                #st.session_state.chat_history.append(ai_msg)

                    
if __name__ == "__main__":
    main()