| Model | AUC | AP | Recall@slots (95% CI) | 2024 recall |
|---|---|---|---|---|
| Rule: fastest updates first | 0.625 | 0.145 | 0.170 (0.144–0.196) | 0.195 |
| Rule: popular genres first | 0.742 | 0.216 | 0.261 (0.239–0.295) | 0.282 |
| Rule: experienced authors first | 0.565 | 0.120 | 0.148 (0.127–0.176) | 0.177 |
| LR: tags + structure | 0.810 | 0.345 | 0.387 (0.359–0.420) | 0.405 |
| LR: + author experience | 0.814 | 0.360 | 0.398 (0.372–0.436) | 0.414 |
| LR: + first-3-chapter text **(main)** | 0.887 | 0.486 | 0.482 (0.455–0.515) | 0.453 |
| LR: chapter text only | 0.856 | 0.402 | 0.406 (0.376–0.440) | 0.377 |
| Boosting: tags + author + text | 0.865 | 0.468 | 0.476 (0.445–0.503) | 0.470 |
| MLP: chapter text only | 0.858 | 0.403 | 0.425 (0.395–0.457) | 0.388 |
| MLP: tags + author + text | 0.886 | 0.493 | 0.493 (0.466–0.527) | 0.456 |
| LR: + prior-book success † | 0.896 | 0.536 | 0.515 (0.491–0.548) | 0.490 |
| LR: + blurb † | 0.918 | 0.629 | 0.573 (0.547–0.603) | 0.566 |
| LR: blurb only † | 0.873 | 0.536 | 0.506 (0.481–0.538) | 0.491 |

† uses information that may postdate chapter 3 (see README); upper bound only. Random = 0.10.
