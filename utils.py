import numpy as np 
import itertools

def get_stoi(file):
  """
  Args:
  file (str): The file path of the repective algnments under data/ folder
  
  Returns:
  vocab (np.array([])): Returns an array Representing the string with indices 
  corresponding to the characters in actual vocabulary.
  """
  vocabulary = "abcdefghijklmnopqrstuvwxyz1234567890!?' "
  f = open(file, "r")
  list = []
  vocab = []
  for line in f:
    line = line.strip()
    txt = line.split(" ")[2]
    if txt != "sil":
        list.append(txt)
        list.append(" ")
  for ls in list:
    for t in ls:
      if t in vocabulary:
          vocab.append(vocabulary.index(t))
  # The length of the sequence is 35
  if len(vocab) < 35:
    for i in range(35-len(vocab)):
      vocab.append(38)
  vocab = np.array(vocab)
  return vocab

def itos(vec):
  """"
  Args:
  vec(np.array([])): An integer array corresponding to the actual character in the vocabulary
  Returns:
  sentence(str): The actual string/sentence
  """
  vocabulary = "abcdefghijklmnopqrstuvwxyz1234567890!?' "
  sentence = ""
  for elem in vec:
    sentence += vocabulary[elem]

  return sentence

def simple_ctc_decode(input_string):
    """
    Its a simple implementation of the actual ctc decode function to 
    convert decode the strings.
    Args:
    input_string(str): The actual string of frame length constant (75)
    Returns:
    decoded_string(str): Returns CTC decoded string.
    """
    # Split the string to process each word separately
    words = input_string.split()
    decoded_words = []

    for word in words:
        # Collapse repeated characters in each word.
        collapsed_word = ''.join(char for char, _ in itertools.groupby(word))
        decoded_words.append(collapsed_word)

    # Rejoin the decoded words into a single string.
    decoded_string = ' '.join(decoded_words)
    return decoded_string

if __file__ == '__main__':

    input_string = "llllllaaaayyyyy ggggrrreeeeennnnnnnn bbblllluee"
    output_string = simple_ctc_decode(input_string)
    print(output_string)  # "lay green blue"