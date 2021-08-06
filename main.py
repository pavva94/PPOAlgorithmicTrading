# coding=utf-8

"""
Goal: Program Main.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse

from tradingSimulator import TradingSimulator

###############################################################################
##################################### MAIN ####################################
###############################################################################

if (__name__ == '__main__'):
    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='PPO', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='Apple', type=str, help="Name of the stock (market)")
    parser.add_argument("-numberOfEpisodes", default=50, type=int, help="Number of training episodes")
    parser.add_argument("-batch_mode", default=False, type=bool, help="Batch Mode for training")
    parser.add_argument("-displayTestbench", default=False, type=bool, help="Dislay Testbench")
    parser.add_argument("-analyseTimeSeries", default=False, type=bool, help="Start Analysis Time Series")
    parser.add_argument("-simulateExistingStrategy", default=False, type=bool, help="Start Simulation of an Existing Strategy")
    parser.add_argument("-evaluateStrategy", default=False, type=bool, help="Start Evaluation of a Strategy")
    parser.add_argument("-evaluateStock", default=False, type=bool, help="Start Evaluation of a Stock")
    args = parser.parse_args()

    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock
    multipleStock = False
    # check if stock are multiple, divided by char '-'
    if '-' in stock:
        stock = stock.split('-')
        print(stock)
        multipleStock = True

    numberOfEpisodes = args.numberOfEpisodes
    batch_mode = args.batch_mode
    displayTestbench = args.displayTestbench
    analyseTimeSeries = args.analyseTimeSeries
    simulateExistingStrategy = args.simulateExistingStrategy
    evaluateStrategy = args.evaluateStrategy
    evaluateStock = args.evaluateStock

    # Training and testing of the trading strategy specified for the stock (market) specified
    if multipleStock:
        simulator.simulateMultipleStrategy(strategy, stock, numberOfEpisodes=numberOfEpisodes, saveStrategy=False)
    else:
        simulator.simulateNewStrategy(strategy, stock, numberOfEpisodes=numberOfEpisodes, batch_mode=batch_mode, saveStrategy=False)

    # the other functions can't be used with multipleStock, so it's used the first of the list
    if displayTestbench:
        simulator.displayTestbench()
    if analyseTimeSeries:
        simulator.analyseTimeSeries(stock[0] if multipleStock else stock)
    if simulateExistingStrategy:
        simulator.simulateExistingStrategy(strategy, stock[0] if multipleStock else stock)
    if evaluateStrategy:
        simulator.evaluateStrategy(strategy, saveStrategy=False)
    if evaluateStock:
        simulator.evaluateStock(stock[0] if multipleStock else stock)
