import sqlite3
import pandas as pd

conn = sqlite3.connect("Chinook_Sqlite.sqlite")
cursor = conn.cursor()

query = '''
SELECT BillingCountry, SUM(Total) as Revenue
FROM Invoice
GROUP BY BillingCountry
ORDER BY Revenue DESC
LIMIT 10;
'''
df = pd.read_sql_query(query, conn)
print(df)


query1 = '''
SELECT c.FirstName, c.LastName, SUM(i.Total) as TotalSpent
FROM Invoice i
JOIN Customer c ON i.CustomerId = c.CustomerId
GROUP BY c.CustomerId
ORDER BY TotalSpent DESC
LIMIT 5;
'''
df1 = pd.read_sql_query(query1, conn)
print("\nTop 5 clients ayant dépensé le plus:")
print(df1)


query2 = '''
SELECT g.Name as Genre, SUM(il.UnitPrice * il.Quantity) as Revenue
FROM InvoiceLine il
JOIN Track t ON il.TrackId = t.TrackId
JOIN Genre g ON t.GenreId = g.GenreId
GROUP BY g.Name
ORDER BY Revenue DESC;
'''
df2 = pd.read_sql_query(query2, conn)
print("\nGenres musicaux rapportant le plus:")
print(df2)


query3 = '''
SELECT AVG(t.Milliseconds / 1000.0) as AvgDurationSeconds
FROM Track t
JOIN Genre g ON t.GenreId = g.GenreId
WHERE g.Name = 'Rock';
'''
df3 = pd.read_sql_query(query3, conn)
print("\nDurée moyenne d’un morceau de Rock (en secondes):")
print(df3)


query4 = '''
SELECT e.FirstName || ' ' || e.LastName as EmployeeName, SUM(i.Total) as Revenue
FROM Employee e
JOIN Customer c ON e.EmployeeId = c.SupportRepId
JOIN Invoice i ON c.CustomerId = i.CustomerId
WHERE e.Title = 'Sales Support Agent'
GROUP BY e.EmployeeId
ORDER BY Revenue DESC
LIMIT 1;
'''
df4 = pd.read_sql_query(query4, conn)
print("\nEmployé (Sales Support Agent) ayant généré le plus de revenus:")
print(df4)

conn.close()