# Class demos
Code demos for the August 2018 Multi-core programming course.

# Configuración para editar archivos usando Sublime

## Configurando multihop SSH

El primer paso es crear un archivo de configuración de ssh.

Abre el archivo con `nano ~/.ssh/config`

    host <nombre que quieras>
    ProxyCommand  ssh <matricula>@ingenieria.csf.itesm.mx nc %h %p
    HostName 10.49.5.130
    User <matricula>
    RemoteForward 52698 localhost:52698

Ahora al usar `ssh <nombre que quieras>` hará los dos saltos directamente.

## Instalar paquete que use rmate

 - [rsub](https://github.com/henrikpersson/rsub) para Sublime
 - [Remote VS Code](https://marketplace.visualstudio.com/items?itemName=rafaelmaiolla.remote-vscode) para VS Code
 - [Remote Atom](https://atom.io/packages/remote-atom) para Atom

## Modificar archivos
Para modificar cualquier archivo, ya habiendo establecido la conexión ssh, utiliza `rmate <archivo>` para modificarlo directamente en tu computadora.

Gracias Omar Sanseviero por el dato.