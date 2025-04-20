These are for 5x5 boards with 4 in a row

The model_43_150_Gomoku_5x5.pt is the checkpoint for a `ResNet(game, 10, 128, device)`. It was trained using num_searches=150.
The model_21_300_Gomoku_5x5.pt is also a checkpoint for a `ResNet(game, 10, 128, device)`. It is a stronger player than the one above (was trained on num_searches=300), but was trained for less iterations.

Feel free to try both.
