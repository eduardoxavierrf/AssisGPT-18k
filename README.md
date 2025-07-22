# Implementação NumPy de um Mini GPT-2

Este repositório apresenta uma implementação educacional de um modelo Transformer do tipo **Decoder-Only**, inspirado no **GPT-2**, utilizando **apenas NumPy**. O objetivo é proporcionar uma compreensão mais profunda do funcionamento interno de um modelo de linguagem neural, camada por camada.

O modelo foi treinado em um corpus baseado nas obras **"Dom Casmurro"**, **"O Alienista"**, **"Memórias Póstumas de Brás Cubas"** e **"Quincas Borba"**, com uma tokenização em nível de caractere (*char-level*), o que permite gerar texto com vocabulário limitado de maneira didática.
## Uso

Primeiramente deve ser feito o treinamento do modelo rodando o arquivo `train.py`.Depois, o modelo pode ser utilizado rodando o arquivo `generate.py` e o contexto pode ser alterado na prompt=.