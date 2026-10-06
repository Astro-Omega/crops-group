# Bienvenido a la documentación inicial de Crops-group

Para empezar es necesario que tengas instalado una version de python 3.10 > actual y del editor de texto ``Visual studio code`` para poder inciar, posteriormente requieres de clonar el repositorio para realizar los siguientes pasos.

### Instalación del entorno virtual
----
Se requiere de un espacio virtualizado como es el caso del ``.venv`` para no manchar tu entorno global, a continuación puede observar los siguientes comandos.


en la consola del VScode escribe
```PowerShell
python -m venv .venv
```
Esta instrucción permite generar el archivo ``.venv``, se recomienda que espere unos minutos hasta que termine de cargar el entorno.

Seguidamente ejecute este comando para activar el entorno virtual
```PowerShell
.venv/Scripts/activate
```

- Para desactivar el entorno virtual ejecute el comando 
```PowerShell
deactivate
```

Seguidamente requiere de instalar las dependencias con el siguiente comando, no es necesario que usted instale de forma manual las dependencias que exige el proyecto.

(ADVERTENCIA): Antes de ejecutar este comando verifique que el entorno virtual este habilitado.

```PowerShell
pip install -r requirements.txt
```

Si descargaste las dependencias en el lugar incorrecto o por casualidades del tiempo olvidaste establecer el entorno virtual, pruebe desinstalando e instalando nuevamente las dependencias usando
```PowerShell
pip uninstall -r requirements.txt -y
```
----
## Ejecución de Crops-group

La ejecución de este algoritmo es propio de la consola shell ``VScode``, por lo tanto se requiere de escribir el siguiente comando

INPORTANTE: Es necesario que se haya realizado los anteriores pasos para evitar problemas con el compilador o ejecución de la misma aplicación

```PowerShell
streamlit run app.py
```

>PD: en los comentario del archivo ``app.py`` se encuentra 
nuevamente la indicación del comando anterior.
Aparte de esos se tiene que aceptar todos los permisos ya que se esta empleando en una API de terceros que te redirige a un servicio en la web
