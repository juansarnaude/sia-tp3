# SIA TP3 : Perceptrones

Gaston Alasia
Juan Segundo Arnaude
Bautista Canevaro
Matias Wodtke

En este trabajo se implementan diversos tipos de perceptrones: simples lineales y no lineales y tambien multicapa.



### Ejercicio 1:

Correr con el siguiente comando desde la carpeta raiz:

```
> python -m src.ejs.ej1
```

El archivo de configuracion se encuentra en `configs/ej1.json`

Aqui un ejemplo de este archivo:

```.json
{
  "input_file": "./input/ej1/and.csv",
  "output_file": "./output/ej1/output.csv",
  "learning_rate": 0.1,
  "periods": 100,
  "epsilon": 0
}
```

`input_file`: Archivo de imput con las entradas de datos.

`output_file`: Archivo de salida con ____ TODO COMPLETAR.

`learning_rate`: tasa de aprendizaje.

`periods`: limite de iteraciones para el algoritmo.

`epsilon`: error aceptado sobre la muestra recibida.

### Ejercicio 2 Linear:

Correr con el siguiente comando desde la carpeta raiz:

```
> python -m src.ejs.ej2
```

El archivo de configuracion se encuentra en `configs/ej2.json`

Aqui un ejemplo de este archivo:

```.json
{
  "input_file": "./input/ej2/ej2.csv",
  "output_file": "./output/ej2/output_linear.csv",
  "learning_rate": 0.0001,
  "periods": 10000,
  "epsilon": 0.0001,
  "k": 2
}
```

`input_file`: Archivo de imput con las entradas de datos.

`output_file`: Archivo de salida con ____ TODO COMPLETAR.

`learning_rate`: tasa de aprendizaje.

`periods`: limite de iteraciones para el algoritmo.

`epsilon`: error aceptado sobre la muestra recibida.

`k`: parametro para k-cross-validation, dividira el input en k partes utilizando k-1 partes como conjunto de entrenamiento y 1 de configuracion de testeo. En caso de desear simplemente entrenar al modelo de neurona y ver la evolucion del error utilizar 'k=1'.


### Ejercicio 2 No Linear:

Correr con el siguiente comando desde la carpeta raiz:

```
> python -m src.ejs.ej2NonLinear
```

El archivo de configuracion se encuentra en `configs/ej2NonLinear.json`

Aqui un ejemplo de este archivo:

```.json
{
  "input_file": "./input/ej2/ej2.csv",
  "output_file": "./output/ej2/output_nonlinear_tanh.csv",
  "learning_rate": 0.0001,
  "periods": 10000,
  "epsilon": 0.0001,
  "k": 1,
  "beta":2,
  "activation": "tanh"
}
```

`input_file`: Archivo de imput con las entradas de datos.

`output_file`: Archivo de salida con ____ TODO COMPLETAR.

`learning_rate`: tasa de aprendizaje.

`periods`: limite de iteraciones para el algoritmo.

`epsilon`: error aceptado sobre la muestra recibida.

`k`: parametro para k-cross-validation, dividira el input en k partes utilizando k-1 partes como conjunto de entrenamiento y 1 de configuracion de testeo. En caso de desear simplemente entrenar al modelo de neurona y ver la evolucion del error utilizar 'k=1'.

`beta`: valor de beta a utilizar en las funciones de activacion.

`activation`: 'tanh' o 'sigmoid'.


### Ejercicio 3 (a,b,c):

Correr con el siguiente comando desde la carpeta raiz:

```
> python -m src.ejs.ej3a
```
```
> python -m src.ejs.ej3b
```
```
> python -m src.ejs.ej3c
```


El archivo de configuracion se encuentra en `configs/ej3a.json`, `configs/ej3b.json`, `configs/ej3c.json` respectivamente

Aqui un ejemplo de este archivo:

```.json
{
  "input_file": "./input/ej3/ej3.txt",
  "output_file": "./output/ej3/output.csv",
  "layer_sizes": [35,10, 1],
  "learning_rate": 0.01,
  "activation_function":"tanh",
  "epochs": 5000,
  "epsilon": 0.01,
  "gaussian_noise":0,
  "beta":1,
  "optimizer": {
    "method":"gradient_descent",
    "momentum":0.9,
    "adam": {
      "beta_1":0.9,
      "beta_2":0.999
    }
  }
}
```

`input_file`: Archivo de imput con las entradas de datos.

`output_file`: Archivo de salida con ____ TODO COMPLETAR.

`layer_sizes`: arreglo donde cada valor representa la cantidad de neuronas en cada capa. La primer capa debe coincidir con el tamaño del input.

`learning_rate`: tasa de aprendizaje.

`activation_function`: 'tanh' o 'sigmoid'.

`epochs`: limite de iteraciones para el algoritmo.

`epsilon`: error aceptado sobre la muestra recibida.

`k`: parametro para k-cross-validation, dividira el input en k partes utilizando k-1 partes como conjunto de entrenamiento y 1 de configuracion de testeo. En caso de desear simplemente entrenar al modelo de neurona y ver la evolucion del error utilizar 'k=1'.

`gaussian_noise`: ruido aplicado sobre los inputs.

`beta`: valor de beta a utilizar en las funciones de activacion.

`optimizer`:

* `method`: metodo de optimizacion elegido, puede ser 'gradient_descent', 'momentum' o 'adam'

* `momentum`: en caso de elegir momentumm, es el valor de entrada de este metodo de optimizacion.

* `adam`: 

* * `beta_1` y `beta_2` : betas utilizados en el metodo de optimizacion adam.


### Ejercicio 4:

Correr con el siguiente comando desde la carpeta raiz:

```
> python -m src.ejs.ej4
```

El archivo de configuracion se encuentra en `configs/ej4.json`

Aqui un ejemplo de este archivo:

```.json
{
    "output_file": "./output/ej3/output.csv",
    "layer_sizes": [784, 256, 128, 10],
    "learning_rate": 0.01,
    "activation_function":"tanh",
    "epochs": 20,
    "epsilon": 0.01,
    "optimizer": {
      "method":"gradient_descent",
      "momentum":0.9,
      "adam": {
        "beta_1":0.9,
        "beta_2":0.999
      }
    }
  }
```

`input_file`: Archivo de imput con las entradas de datos.

`output_file`: Archivo de salida con ____ TODO COMPLETAR.

`layer_sizes`: arreglo donde cada valor representa la cantidad de neuronas en cada capa. La primer capa debe coincidir con el tamaño del input (en este caso 784).

`learning_rate`: tasa de aprendizaje.

`activation_function`: 'tanh' o 'sigmoid'.

`epochs`: limite de iteraciones para el algoritmo.

`epsilon`: error aceptado sobre la muestra recibida.

`optimizer`:

* `method`: metodo de optimizacion elegido, puede ser 'gradient_descent', 'momentum' o 'adam'

* `momentum`: en caso de elegir momentumm, es el valor de entrada de este metodo de optimizacion.

* `adam`: 

* * `beta_1` y `beta_2` : betas utilizados en el metodo de optimizacion adam.





