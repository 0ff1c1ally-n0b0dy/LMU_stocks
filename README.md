# LMU_stocks
LMU is a neural netowrk similar to LSTM, but newer. It is called Legendre memory unit. 
Please see the following paper for more details: https://arxiv.org/pdf/2102.11417.pdf
I applied it and trained it to identify certain events in the stock market. 
* backtest is the training of the LMU on those events labeled by a human.
* forwardtest is running the LMU as if it was trading live on data that it has never seen before (it decides when to buy/sell etc.)
* backtest_and_forwardtest is a combination of backtest and forwardtest.
* live_trading is the LMU trading live placing trades using INteractive brokers API.
