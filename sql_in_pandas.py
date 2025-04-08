import pandas as pd
import sqlite3 
conn = sqlite3.connect("Chinook_Sqlite.sqlite") 
cursor = conn.cursor() 

query1 = '''
SELECT c.FirstName || ' ' || c.LastName AS Client,
       SUM(i.Total) AS TotalSpent
FROM Customer c
JOIN Invoice i ON c.CustomerId = i.CustomerId
GROUP BY c.CustomerId
ORDER BY TotalSpent DESC
LIMIT 5;
'''
df1 = pd.read_sql_query(query1, conn)
print(df1)

query2 = '''
SELECT g.Name AS Genre,
       SUM(il.UnitPrice * il.Quantity) AS Revenue
FROM Genre g
JOIN Track t ON g.GenreId = t.GenreId
JOIN InvoiceLine il ON t.TrackId = il.TrackId
GROUP BY g.GenreId
ORDER BY Revenue DESC;
'''
df2 = pd.read_sql_query(query2, conn)
print(df2)

query3 = '''
SELECT g.Name AS Genre,
       ROUND(AVG(t.Milliseconds) / 60000, 2) AS AvgDuration_Minutes
FROM Track t
JOIN Genre g ON t.GenreId = g.GenreId
WHERE g.Name = 'Rock';
'''
df3 = pd.read_sql_query(query3, conn)
print(df3)

query4 = '''
SELECT e.FirstName || ' ' || e.LastName AS Employee,
       SUM(i.Total) AS TotalRevenue
FROM Employee e
JOIN Customer c ON e.EmployeeId = c.SupportRepId
JOIN Invoice i ON c.CustomerId = i.CustomerId
WHERE e.Title = 'Sales Support Agent'
GROUP BY e.EmployeeId
ORDER BY TotalRevenue DESC
LIMIT 1;
'''
df4 = pd.read_sql_query(query4, conn)
print(df4)

conn.close() 