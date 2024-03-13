# How to use this repo:

pip install -r requirements.txt

github_url = 'path to github repo'
unixcoder_path = 'path to unixcoder fine-tuned'
using_mistral = 'wants to use mistral? yes or no'
query = 'query to find the similarity in the github'

Example: 
python main.py --github_url "https://github.com/microsoft/autogen/" --model_path '/content/drive/My Drive/unixcoder-ft.bin' --mistral 'yes' --nl_query 'In AutoGen, a conversable agent is an entity with a specific role that can pass messages to send and receive information to and from other conversable agents, e.g., to start or continue a conversation.'


## Para que funcione pdf to json: 
   Linux:
   - wget https://github.com/kermitt2/grobid/archive/0.7.2.zip
   - unzip 0.7.2.zip
   - cd grobid-0.7.2/
   - ./gradlew clean install



Descargan la carpeta completa y ejecutan la siguiente linea en la terminal:

python --github_url "https://github.com/microsoft/autogen/" --pdf_path "/autogen.pdf" --model_path 'unixcoder-ft.bin' --nl_query 'need to know the loss function of seq2seq model'

Esto devuelve las 3 respuestas más similares a la query. Pueden probar modificando los argumentos.

# Qué significa cada archivo?
   - build.py: Lo usé para construir el "parseador". Técnicamente, en linux les deberia funcionar my-languages.so y en windows my-languages1.so. En caso de que ninguno funcione van a necesitar correr este archivo. Si es asi, me hablan y les explico cómo.
     Suele tirar un error del estilo: OSError: [WinError 193] %1 is not a valid Win32 application
   - DFG.py: Se usa para pasar de código "string" a código secuencia.
   - code_search.py: Se usa para cargar el modelo y encontrar los embeddings más similares a la query.
   - extract_code.py: Convierte un repo de github a json.
   - pdf_to_json.py: Convierte un pdf a json.
   - main.py: Agarra el pdf y github y los convierte a JSON. *Importante:* Falta chunkear el json del pdf y generar los embeddings de los chunks, además falta generar los embeddings del código (ya está truncado).
   - my-languages.so: Parseador de linux.
   - my-languages1.so: Parseador de windows.
   - tokens.py: Se usa para pasar de código "string" a código secuencia.
   - utils.py: Funciones varias



  
