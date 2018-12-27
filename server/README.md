# Jubatus Server Examples
Examples of server configuration files that correspond to jubatus client programs.

# How to Deploy
  * Build a docker image from Dockerfile and run.  
    * An example (type following command in a directory which Dockerfile exists.):
  
    ```
    docker build -t jubaclassifier:0.0.1 .
    docker run -d -p 9199:9199 --name 'jubaclassifier' --init jubaclassifier:0.0.1
    ```
  
  * Alternatively, send configuration files to existing jubatus server.
