the code for a custom task in ground truth for instance segmentation
in this case, for 3 classes: person, animal, and inanimate

```
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<crowd-form>
  <crowd-instance-segmentation
    name="annotatedResult"
    src="{{ task.input['source-ref'] | grant_read_access }}"
    header="Segment all instances of person, animal, and inanimate"
    labels="['person','animal','inanimate']"
  >
    <full-instructions header="Segmentation Instructions">
      <ol>
        <li>Inspect the image.</li>
        <li>Select a label and draw masks for every visible instance of that label.</li>
        <li>Repeat until all instances are labeled.</li>
      </ol>
    </full-instructions>

    <short-instructions>
      <p>Create a mask for each instance of person, animal, and inanimate.</p>
    </short-instructions>
  </crowd-instance-segmentation>
</crowd-form>

```