<div id="top"></div>

<h3 align="center">Wikipedia Search Engine Project</h3>

  <p align="center">
    As part of the "Information Retrival" course in Ben-Gurion University of the Negev
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
To properly learn about information retrival and get hands on experience on the topic, as part of this course we have build a working search engine based on the raw wikipedia documents dumps
Using inverted index, we have implemented different search methods using cosine similary of the body, binary search on the titles and anchors.
Our main search method ended up being a mixture of BM25 search with considiration to the page's view count resulting in average 3 seconds retrival time per query and MAP@40 score of 0.655
<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Nltk](https://www.nltk.org/)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/)
* [Math](https://docs.python.org/3/library/math.html)
* [Numpy](https://numpy.org/)
* [Re](https://docs.python.org/3/library/re.html)
* [Collections](https://docs.python.org/3/library/collections.html)
* [Itertools](https://docs.python.org/3/library/itertools.html)
* [Pathlib](https://docs.python.org/3/library/pathlib.html)
* [GoogleCloudStorage](https://cloud.google.com/storage)
* [Ngrok](https://ngrok.com/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

We have made all the index (see in the bucket details) in our storage bucket on the cloud platform.
After that we have created the local instance of the VM machine and connected to it with SSH.
Coppying loccaly all the files from our storage bucket to our instance ensured us fast retrival time.

The 3 files "inverted_index_gcp_anchor_with_dl" "inverted_index_gcp_title_with_dl" and "inverted_index_gcp_with_dl" were the files used to created the corresponding index on Google Cloud Platform ulitizing the DataProc and Google Storage.

"queries_tests" was used for local testing of the search engine, pinging the search server with queries and messuring the results, providing a readable and informative summy of the scores of every test set (30 queries, see the next file).

"queries_train" is the Json file that contained all the queries that we used in training our search models to optimized the retrival time and the retrival quality.

"run_frontend_in_collan" was used to locally test our search engine on Ngrok before setting up the computing engine instance on VM in the Google Cloud platform

"run_frontend_in_gcp-test" was used to set up the spoken abouve coputing engine instance to configure and run it

"search_frontend" is the main search engine heart that loads the indexes, gets the queries from the URL and preforms a search with our chosen method (BM25 search with page view consideration) before returning the results for display.

"startup_script_gcp" was used to install python pip and the majority of the required packages on the computing engine instance.

<!-- USAGE EXAMPLES -->
## Usage

We have used the providded staff skeleton for tests and improved upon it. 
Instead of just sending requests to the server and getting the results, we have processed the results to show the average retrival time per query and the MAP@40 score, as well as how many queries have passed or failed.
While we acknowledge the staff will test differently, we have found it to be very helpfull to messure our engine performance during the experimentation period to find the best search methods.

The main work cycle is :

* Get query from the URL
* Process the query, tokenize it using Re and Nltk
* Pass the token list to the funcion
* Calculate the BM25 score for the given query on the body index
* Use the page views scores and the BM25 scores to get the real score
* After sorting those scores, return up to 100 results back

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License.

Copyright 2022 [arie478](https://github.com/arie478)  [adamuziel](https://github.com/adamuziel)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Arie Kat - ariekat@post.bgu.ac.il

Adam Uziel - adamuziel@post.bgu.ac.il

Project Link: [https://github.com/arie478/ir_proj](https://github.com/arie478/ir_proj)

<p align="right">(<a href="#top">back to top</a>)</p>
