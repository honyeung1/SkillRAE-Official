You are a Python technical documentation engineer responsible for maintaining your team's internal knowledge base. You need to build an automated pipeline to collect Python built-in function information from the official documentation and generate standardized reports.

Please complete the following tasks:

1. Visit the Python official documentation's built-in functions page (https://docs.python.org/3/library/functions.html) and extract the name and description (first sentence of the docstring) of every built-in function listed on the page.

2. Generate builtin_functions.json as a JSON object with the following fields:
   - "source": the URL of the page you scraped (string)
   - "total_count": the total number of functions collected (integer)
   - "functions": an array of objects, each containing "name" (function name) and "description" (brief description), sorted alphabetically by name

3. Generate functions_catalog.xlsx with two worksheets:
   - Sheet 1 "Function Catalog": three columns (No., Function Name, Description), sorted alphabetically, with bold header row and colored background
   - Sheet 2 "Category Statistics": group functions by functional category (e.g., Type Conversion, Math Operations, Sequence Operations, I/O, Object & Class Inspection, Iterator Tools, etc.), showing function count per category and listing all function names under each category

4. Generate catalog_report.docx containing:
   - A report title
   - An overview paragraph (stating the data source and total number of functions collected)
   - Detailed sections by functional category (each category as a Heading 2, listing functions with brief explanations)
   - A summary paragraph (overall assessment of Python's built-in function ecosystem and usage recommendations)