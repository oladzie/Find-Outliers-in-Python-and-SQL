-- Explore the Data 

SELECT * FROM traffic

-- Find the Outlier Days

SELECT 
Date,
Sessions,
(Sessions - AVG(Sessions) over()) / STDEV(Sessions) over() AS zscore

FROM traffic

-- Extreme Outliers

SELECT * FROM
(SELECT 
Date,
Sessions,
(Sessions - AVG(Sessions) over()) / STDEV(Sessions) over() AS zscore

FROM traffic) AS score_table
WHERE zscore > 2.576 OR zscore < -2.576

-- Outlier 2 STD and above/below mean

SELECT * FROM
(SELECT 
Date,
Sessions,
(Sessions - AVG(Sessions) over()) / STDEV(Sessions) over() AS zscore

FROM traffic) AS score_table
WHERE zscore > 1.96 OR zscore < -1.96

-- Outlier 1 STD and above/below mean

SELECT * FROM
(SELECT 
Date,
Sessions,
(Sessions - AVG(Sessions) over()) / STDEV(Sessions) over() AS zscore

FROM traffic) AS score_table
WHERE zscore > 1.645 OR zscore < -1.645