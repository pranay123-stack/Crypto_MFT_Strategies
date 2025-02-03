import csv

# Assuming the CSV data is stored in a file named 'trades.csv'
csv_file = '/Users/pranaygaurav/Downloads/AlgoTrading/company_test/Blockchain_intelligence/trade_history.csv'

# Initialize total profit and initial investment
total_profit = 0
initial_investment = 10000  # Assuming the initial investment is $10,000,000

# Read the CSV file and calculate total profit
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        # Check if the row contains "Profit"
        if len(row) > 0 and "Profit" in row[0]:
            # Extract the profit value and add it to the total
            try:
                profit_loss = float(row[0].split(":")[1].strip())
                total_profit += profit_loss
            except ValueError:
                print("Error parsing profit value:", row[0])

# Calculate percentage return on investment
percentage_return = (total_profit / initial_investment) * 100


# Save results to a new CSV file named 'pnl.csv'
output_file = '/Users/pranaygaurav/Downloads/AlgoTrading/company_test/Blockchain_intelligence/pnl.csv'
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write headers
    csv_writer.writerow(['Total Profit', 'Percentage Return'])
    # Write data
    csv_writer.writerow([total_profit, round(percentage_return, 2)])

# Print the results
print("Results saved to", output_file)
