# CientificGPT:

Para este trabajo buscamos generar un research-assistant que pueda ayudar a los investigadores (o curiosos sobre algún tema en particular) a encontrar la similitudes entre paper y repositorio. Para ello, generamos un modelo que, dado un repositorio de github y una query, devuelve las funciones más similares a la query pedida.

¿Cómo generamos este modelo?

Primero, parseamos el repositorio de github a un archivo json manteniendo la estructura jerárquica de este. Para convertir el código en secuencia de strings seguimos el procedimiento comentado en el paper de UnixCoder, donde proponen una función que devuelve esta secuencia buscada a partir del AST del código. Luego generamos los embeddings de estas funciones usando unixcoder finetuneado.

 **Aclaración**: Desde Microsoft subieron UnixCoder base y el código para finetunear para cada una de las tareas presentadas en el paper. De nuestro parte, finetuneamos con su código para la tarea Code Search -from paper: *The task aims to find the most relevant code from a collection of candidates given a natural language query.*-

Para finalizar con el pipeline, dada una query se busca calcular las funciones más similares a ésta dentro del repositorio. Se puede elegir o no utilizar a Mistral finetuneado para que acerque esta query a docstring antes de pedirle la similaridad a Unixcoder. 


# Qué significa cada archivo?
   - build.py: Lo usamos para construir el "parseador". Técnicamente, en linux les deberia funcionar my-languages.so y en windows my-languages1.so. En caso de que ninguno funcione van a necesitar correr este archivo. Si es asi, me hablan y les explico cómo.
   - my-languages.so: Parseador de linux.
   - my-languages1.so: Parseador de windows.
   - DFG.py: Se usa para pasar de código "string" a código secuencia.
   - code_search.py: Generamos los embeddings y los guardamos en una base de datos FAISS. 
   - extract_code.py: Convierte un repo de github a json.
   - main.py: Dado un github lo convierte a JSON para luego calcular los embeddings de cada una de las funciones del repositorio. Estos embeddings se guardan en una base de FAISS. 
   - tokens.py: Se usa para pasar de código "string" a código secuencia.
   - utils.py: Funciones varias
   - pdf_to_json.py: Convierte un pdf a json.
   - calling_mistral.py: Este script carga el modelo que finetuneamos para acercar las querys a docstring y, dada una query, devuelve el docstring para esta. Se puede decidir usarlo o no.

# Cómo usarlo?


!pip install -r requirements.txt

!pip install -i https://pypi.org/simple/ bitsandbytes

!pip install accelerate

!pip install bitsandbytes

!pip install -q -U git+https://github.com/huggingface/transformers.git@main

!pip install -q -U git+https://github.com/huggingface/peft.git

!pip install -q -U datasets scipy ipywidgets matplotlib


github_url = 'path to github repo'
unixcoder_path = 'path to unixcoder fine-tuned'
mistral = 'wants to use mistral? yes or no'

!python main.py --github_url github_url --model_path unixcoder_path --mistral 'yes'

Ejemplo: 

     ```
     python main.py --github_url "https://github.com/ankitapasad/layerwise-analysis.git" --model_path '/content/drive/My Drive/unixcoder-ft.bin' --mistral 'yes'
     ```

Genera una conversación con un asistente al cual se le puede ir preguntando sobre el github para que devuelta las funciones más similares a la query.



## Para que funcione pdf to json: 
   Linux:
   - wget https://github.com/kermitt2/grobid/archive/0.7.2.zip
   - unzip 0.7.2.zip
   - cd grobid-0.7.2/
   - ./gradlew clean install



  
