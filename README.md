@ https://www.reddit.com/user/yoonboBOT

# AskReddit Generator Bot by Yoonbo Cho
I generate Askreddit questions using TensorFlow!

## Examples:
https://www.reddit.com/r/AskReddit/comments/j8ux4r/whats_reddit_what_is_the_side_effects_covid_you/
```
Whats reddit what is the side effects covid you which was to the many?
```
https://www.reddit.com/r/AskReddit/comments/jf0qvd/what_is_for_your_you_how_theory_10_to_thing_do/
```
What is for your you how theory 10 to thing do faking it reddit?
```

## How Do I Work?
<ul>
  <li>My <a href="https://github.com/ProHanzo/AskReddit/blob/master/Code/ETL.py">ETL.py</a> program handles the <b>BackEnd</b> of this Bot:
    <ul>
      <li>I extract data from AskReddit using a Reddit API. </li>
      <li>I merge the new data to the old data from github, removing duplicates. </li>
      <li>Using the new dataset, I train a Classification and Generation Model using TensorFlow. </li>
      <li>I save and load the new machine learning models and the new merged dataset to github.</li>
    </ul>
  </li>
</ul>
<ul>
  <li>My <a href="https://github.com/ProHanzo/AskReddit/blob/master/Code/ETL.py">Bot.py</a> program handles the <b>FrontEnd</b> of this Bot:
    <div></div>
    <ul>
      <li>I extract the trained machine learning models from github. </li>
      <li>I generate n number of Askreddit question using the Generation Model.</li>
      <li>I classify the list of questions depending on their quality.</li>
      <li>I pick the best question then post that question on Askreddit using this Bot.</li>
    </ul>
  </li>
</ul>
  
