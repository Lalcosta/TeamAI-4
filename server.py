from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load
from werkzeug.utils import secure_filename
import os

#Cargar el modelo
dt = load('FinalModel.joblib')

#Generar el servidor (Back-end)
servidorWeb = Flask(__name__)


@servidorWeb.route("/formulario",methods=['GET'])
def formulario():
    return render_template('formulario.html')

#Envio de datos a través de Archivos
@servidorWeb.route('/modeloFile', methods=['POST'])
def modeloFile():
    f = request.files['file']
    filename=secure_filename(f.filename)
    path=os.path.join(os.getcwd(),'files',filename)
    f.save(path)
    file = open(path, "r")
    
    for x in file:
        info=x.split()
    print(info)
    datosEntrada = np.array([
            int(info[0]),
            int(info[1]),
            float(info[2]),
            float(info[3]),
            float(info[4]),
            float(info[5]),
            float(info[6]),
            float(info[7]),
            int(info[8]),
            int(info[9]),
            int(info[10]),
            int(info[11]),
            int(info[12])
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

#Envio de datos a través de Forms
@servidorWeb.route('/modeloForm', methods=['POST'])
def modeloForm():
    #Procesar datos de entrada 
    contenido = request.form
    
    datosEntrada = np.array([
            contenido['hp'],
            contenido['dest'],
            contenido['age'],
            contenido['rs'],
            contenido['fc'],
            contenido['sm'],
            contenido['spa'],
            contenido['vrd'],
            contenido['cd'],
            contenido['cn'],
            contenido['cs'],
            contenido['cryo'],
            contenido['vip']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})


#Envio de datos a través de JSON
@servidorWeb.route('/modelo', methods=['POST'])
def modelo():
    #Procesar datos de entrada 
    contenido = request.json
    print(contenido)
    datosEntrada = np.array([
            contenido['hp'],
            contenido['dest'],
            contenido['age'],
            contenido['rs'],
            contenido['fc'],
            contenido['sm'],
            contenido['spa'],
            contenido['vrd'],
            contenido['cd'],
            contenido['cn'],
            contenido['cs'],
            contenido['cryo'],
            contenido['vip']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})



@servidorWeb.route("/app",methods=["POST"])
def app():
    data=request.json
    arr=[]
    print(data)
    for i in data:
        arr.append(data[i])
        print(i,":",data[i])
    print(arr)
    return arr


if __name__ == '__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8080')