The MDF Python API can be used to create or load an MDF model for inspection and validation.
It also includes a basic execution engine for simulating models in the format. See the HTML documentation
for these modules [here](https://mdf.readthedocs.io/en/latest/api/_autosummary/modeci_mdf.html)

<table class="longtable docutils align-default">
<colgroup>
<col style="width: 10%">
<col style="width: 90%">
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="modeci_mdf.execution_engine.html#module-modeci_mdf.execution_engine" title="modeci_mdf.execution_engine"><code class="xref py py-obj docutils literal notranslate"><span class="pre">modeci_mdf.execution_engine</span></code></a></p></td>
<td><p>The reference implementation of the MDF execution engine; allows for executing <code class="xref py py-class docutils literal notranslate"><span class="pre">Graph</span></code> objects in Python.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="modeci_mdf.functions.html#module-modeci_mdf.functions" title="modeci_mdf.functions"><code class="xref py py-obj docutils literal notranslate"><span class="pre">modeci_mdf.functions</span></code></a></p></td>
<td><p>Specifies and implements the MDF the function ontology; a collection of builtin functions that can be used in MDF <code class="xref py py-class docutils literal notranslate"><span class="pre">Function</span></code> and <code class="xref py py-class docutils literal notranslate"><span class="pre">Parameter</span></code> objects.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="modeci_mdf.interfaces.html#module-modeci_mdf.interfaces" title="modeci_mdf.interfaces"><code class="xref py py-obj docutils literal notranslate"><span class="pre">modeci_mdf.interfaces</span></code></a></p></td>
<td><p>Implementations of importers and exporters for supported environments; fulfilling the <a class="reference external" href="https://github.com/ModECI/MDF/tree/main/examples">hub and spoke model</a> of MDF by allowing exchange between different modeling environments via MDF.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="modeci_mdf.mdf.html#module-modeci_mdf.mdf" title="modeci_mdf.mdf"><code class="xref py py-obj docutils literal notranslate"><span class="pre">modeci_mdf.mdf</span></code></a></p></td>
<td><p>The main object-oriented implementation of the MDF schema, with each core component of the <a class="reference external" href="../Specification.html">MDF specification</a> implemented as a <code class="code docutils literal notranslate"><span class="pre">class</span></code>.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="modeci_mdf.utils.html#module-modeci_mdf.utils" title="modeci_mdf.utils"><code class="xref py py-obj docutils literal notranslate"><span class="pre">modeci_mdf.utils</span></code></a></p></td>
<td><p>Useful utility functions for dealing with MDF objects.</p></td>
</tr>
</tbody>
</table>
