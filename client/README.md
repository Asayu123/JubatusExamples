# Jubatus Client Examples
Examples of standalone programs that connects to jubatus server.

# Directory structures
* Create a directory under ./main per one theme like following.

```bash
 .
├── README.md
├── main
│       ├── mail_classifier
│       │       ├── predict.py
│       │       └── train.py
│       └── mail_recommender
│                  ├── predict.py
│                  └── train.py
└── modules
           └── wrapper.py
```

# How to Run
1. Add a path of this directory to $PYTHONPATH from **project root** before you run.

    ```bash
    export PYTHONPATH="./client:$PYTHONPATH"`
    ```

2. Type following command from **project root**.

    ```bash
    python ./client/main/<theme_name>/<script_name>.py
    ```
