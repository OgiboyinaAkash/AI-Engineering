**LLM output generation**
- LLMs at their core are next-token predictors.
- LLM takes user input and converts them into tokens (Easy to work on token ids than direct text)
- These tokens are passed through a neural network (a transformer).
- It predicts a probability distribution over possible next tokens
- It picks one token, appends it, and repeats

**Context Window**
- model’s short-term memory.
- It’s the maximum number of tokens the model can consider at once
- Includes:
    - prompt
    - previous conversations
    - its own generated output
- Old tokens get dropped when you exceed the context window limit

**Temperature** 
- It controls randomness in token selection.
- Low Temperature (0.1-0.3)
    - picks highest probability tokens
    - more deterministic
- Medium Temperature (~0.7)
    - balanced creativity
- High Temperature (>1)
    - higher chance of unusual or incorrect outputs

**Non determinism**
- Even with same input LLM can generate differnt output
- As model samples from a probability distribution
- 2 tokens may have same probability 
- a differnt token can affect the next tokens as the probabilities differ
- thus produces a differnt output

**RAG**
- Problem : LLMs tend to hallucinate and give black box reasoning

- Solution : Retrieve relevant information from querying external sources —> Augment with pre-trained knowledge —> Generate response
    - R-Retrieval
    - A-Augmentation
    - G-Generation
 
- **Data indexing**
    - Extract text from documents and transform them into vector embeddings. 
    - This preserves semantic meaning by assigning similar embedding values to co-related words. 
    - The words king and queen are correlated, so they both have similar vectors. It forms our retrieval Database.

- **Data Retrieval and Generation**
    - The user's query is transformed into an embedding using the same model as earlier
    - Now Euclidean distance is calculated with the vectors in the database, and the closest ones are chosen to augment with the pretrained knowledge to generate the response.

**Fine Tuning**
- A transfer learning technique 
    - Model developed for one task is reused as a starting point for a model on second related task.
    - It leverages knowlegedge like weights, feautures and patterns gained from solving one problem to improve generalization on another.
- Pre-trained model instead of training from scratch just trained further on a smaller, specialized dataset to adapt for a specific task.

**Chunking**
- preprocessing technique of breaking large documents into smaller, manageable segments
- Effective chunking balances semantic coherence keeping relevant information together with performance constraints

    - **Fixed Size chunking:** T
        - Technique that splits text into uniform, predetermined segments based on token counts. 
        - Simple, fast, and ensures input fits within model context windows

    - **Sentence aware chunking:** 
        - Technique used to split long text into smaller segments by respecting sentence boundaries.
        - Instead of cutting text abruptly after a fixed number of characters or words, this method identifies sentence-ending punctuation to ensure that each chunk contains one or more complete sentences.
    - **Semantic chunking:**
        - Text-segmentation technique that divides documents based on meaning—rather than fixed character counts—by detecting topic shifts between sentences, often using vector embeddings.
        - It creates highly coherent, context-aware chunks, making retrieved context is semantically relevant.

**Neural Networks**
- The first layer will be an embedding layer where these text are converted into a vector using a embedding model (Ex. Word2Vec)
- the ouput of this layer will be passed to the next layer which is the first layer of hidden layers.
- the outputs of each hidden layer will be passed as an input of next hidden layer.
- at last there is an ouput layer which genrates our output