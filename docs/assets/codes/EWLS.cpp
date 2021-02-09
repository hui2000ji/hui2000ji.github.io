#include<vector>
#include<set>
#include<algorithm>
#include<ctime>
#include<cmath>
#include<cstring>
#include<cstdio>
#include<cstdlib>
#define DEBUG
// #define VERBOSE
using namespace std;

class Edge
{
public:
	int x;
	int y;
	int age;
	Edge(int _x, int _y, int _age = 0): age(_age) 
	{
		if (_x > _y)
		{
			swap(_x, _y);
		}
		x = _x;
		y = _y;
	}
	bool operator<(const Edge& other) const
	{
		if (age == other.age)
		{
			if (x == other.x)
				return y < other.y;
			else
				return x < other.x;
		}
		else
			return age < other.age;
	}
};

const int N = 751,           // maximum number of points + 1
          Delta = 1,         // number of nodes to remove when new upper bound is found
          max_cand = 2,     // max number of candidate nodes to sample from when finding exchangable pairs
          steps = 5000;     // max steps to run
int w[N][N];                 // edge weights: an int matrix
vector<int> edge[N];         // adjacency list: an array of int vectors
bool C[N];                   // candidate set: a bool array
bool C_star[N];              // best candidate set: a bool array
set<Edge> L;                 // set of uncovered edges
set<Edge> UL;                // set of locally unvisited uncovered edges
int dscore[N];               // dscore of each node: an int array
int n,                       // number of nodes
    m,                       // number of edges
    ub,                      // upper bound on maximum vertex cover
	traversal_order[N]{};    // a utility array for node traversal


/*
Get the size of the vertex cover (candidate) C.
Implemented by a sum over the boolean vector C.
*/
inline int get_vc_size(bool* C)
{
	int ans = 0;
	for (int i = 0; i < n; i++)
		ans += C[i];
	return ans;
}

/*
Make the current vertex cover candidate C a valid vertex cover using a greedy algorithm.
Arguments:
    C: current vertex cover candidate. Will be a valid vertex cover after calling this function.
    _e: current uncovered edge.
    prefer_small_node_id: If true, will add the node with smaller node id to C when covering the uncovered edges.
*/
int greedy_mvc_solver(bool* C, const set<Edge>& _e)
{
	set<Edge> e = _e;
	for (auto i = e.begin(); i != e.end(); i = e.erase(i))
	{
		if (!C[i->x] && !C[i->y])
		{
			// if (prefer_small_node_id)
			if (rand() % 2)
				C[i->x] = true;
			else
				C[i->y] = true;
		}
		
	}
	return get_vc_size(C);
}

/*
Compute the d_score for node u.
*/
inline int get_dscore(const int u)
{
	int ans = 0;
	for (int v : edge[u])
	{
		if (!C[v])
			ans += w[u][v];
	}
	if (C[u])
		ans = -ans;
	return ans;
}

/*
Update the d_scores after node u is inserted to / deleted from the vertex cover candidate C.
*/
void update_dscore(const int u, const bool is_insert)
{
#ifdef VERBOSE
	printf("update %d is_insert = %b\n", u, is_insert);
#endif
	dscore[u] = get_dscore(u);
	for (int i : edge[u])
	{
		dscore[i] = get_dscore(i);
	}
}

/*
Remove delta nodes from the current vertex color candidate C.
*/
void remove_delta(const int &delta, const int &step)
{
	for (int remove_cnt = 0; remove_cnt < delta; remove_cnt++)
	{
		int max_dscore = 0x80000000;
		int index = -1;
		for (int i = 0; i < n; i++)
		{
			if (C[i] && dscore[i] > max_dscore)
			{
				max_dscore = dscore[i];
				index = i;
			}
		}
		if (index == -1)
			break;
		C[index] = false;
		update_dscore(index, false);
	}
	// build L UL
	for (int i = 0; i < n; i++)
	{
		for (int j : edge[i])
		{
			if (j > i && !C[i] && !C[j])
				L.insert(Edge(i, j, step));
		}
	}
	UL = L;
}

/*
Compute the score of swapping nodes u and v, where one of them is in C and the other is not.
*/
inline int score(const int u, const int v)
{
	return dscore[u] + dscore[v] + w[u][v];
}

/*
Find exchangeble node pairs from L and UL.
*/
pair<int, int> find_exchangable_pair()
{
	set<Edge>::iterator si = L.begin();  // si->x and si->y are both out of C
	if (si != L.end())
	{
		int cand_nodes_x[max_cand]{}, cand_nodes_y[max_cand]{}, ix = 0, iy = 0;
		for (int u : traversal_order)
		{
			if (!C[u])
				continue;
			if (score(u, si->x) > 0)
				cand_nodes_x[ix++] = u;
			if (score(u, si->y) > 0)
				cand_nodes_y[iy++] = u;
			if (ix + iy >= max_cand)
				break;
		}
		if (ix + iy > 0)
		{
			int rand_idx = rand() % (ix + iy), u, v;
			if (rand_idx < ix) {
				u = cand_nodes_x[rand_idx];
				v = si->x;
			}
			else {
				u = cand_nodes_y[rand_idx - ix];
				v = si->y;
			}
			UL.erase(*si);
			L.erase(si);
			return make_pair(u, v);
		}
	}

	
	for (si = UL.begin(); si != UL.end(); si = UL.erase(si))
	{
		int cand_nodes_x[max_cand]{}, cand_nodes_y[max_cand]{}, ix = 0, iy = 0;
		for (int u : traversal_order)
		{
			if (!C[u])
				continue;
			if (score(u, si->x) > 0)
				cand_nodes_x[ix++] = u;
			if (score(u, si->y) > 0)
				cand_nodes_y[iy++] = u;
			if (ix + iy >= max_cand)
				break;
		}
		if (ix + iy > 0)
		{
			int rand_idx = rand() % (ix + iy), u, v;
			if (rand_idx < ix) {
				u = cand_nodes_x[rand_idx];
				v = si->x;
			}
			else {
				u = cand_nodes_y[rand_idx - ix];
				v = si->y;
			}
			UL.erase(si);
			return make_pair(u, v);
		}
	}

	return make_pair(0, 0);
}

/*
Insert node v to the current vertex cover candidate C.
d_score, L and UL will be updated accordingly.
*/
void insert_to_vertex_cover(int v)
{
	C[v] = 1;
	update_dscore(v, true);
	for (set<Edge>::iterator si = L.begin(); si != L.end();)
	{
		if (si->x == v || si->y == v)
			si = L.erase(si);
		else
			si++;
	}
	for (set<Edge>::iterator si = UL.begin(); si != UL.end();)
	{
		if (si->x == v || si->y == v)
			si = UL.erase(si);
		else
			si++;
	}
}

/*
Delete node v from the current vertex cover candidate C.
d_score, L and UL will be updated accordingly.

*/
void delete_from_vertex_cover(int u, int step)
{
	C[u] = 0;
	update_dscore(u, false);
	for (int i : edge[u])
	{
		if (!C[i])
		{
			L.insert(Edge(u, i, step));
			UL.insert(Edge(u, i, step));
		}
	}
}

/*
Initialize the graph from stdin input.
*/
void init()
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			w[i][j] = 1;
	for (int i = 0; i < n; i++)
	{
		traversal_order[i] = i;
		C[i] = C_star[i] = dscore[i] = 0;
		edge[i].clear();
	}
	L.clear();
	UL.clear();
	// input edges
	for (int i = 0; i < m; i++)
	{
		int u, v;  // edge endpoints
		scanf("%d%d", &u, &v);
		u--;
		v--;
		w[u][v] = 0;
		w[v][u] = 0;
	}
	// add self loop to ensure no self-loop present in complement graph
	for (int i = 0; i < N; i++)
		w[i][i] = 0;
	// construct complement graph
	for (int i = 0; i < n; i++)
	{
		for (int j = i + 1; j < n; j++)
		{
			if (w[i][j] == 1)
			{
				edge[i].push_back(j);
				edge[j].push_back(i);
				L.insert(Edge(i, j));
			}
		}
	}
	ub = greedy_mvc_solver(C, L);
	L.clear();
	memcpy(C_star, C, n);
	for (int i = 0; i < n; i++)
		dscore[i] = get_dscore(i);
	remove_delta(Delta, 0);
}

/*
The main function of the EWLS algorithm.
*/
void EWLS(const int max_step)
{
	init();
	for (int step = 1; step < max_step; ++step)
	{
		if (step % 20 == 0)
			random_shuffle(traversal_order, traversal_order + n);

		pair<int, int> ii = find_exchangable_pair();
		if (ii != make_pair(0, 0))
		{
			int u = ii.first, v = ii.second;
			// C = C - u + v
			delete_from_vertex_cover(u, step);
			insert_to_vertex_cover(v);
		}
		else
		{
			for (set<Edge>::iterator si = L.begin(); si != L.end(); si++)
			{
				w[si->x][si->y]++;
				w[si->y][si->x]++;
			}
			for (int i = 0; i < n; i++)
				dscore[i] = get_dscore(i);
			vector<int> inside, outside;
			for (int i = 0; i < n; i++)
			{
				if (C[i])
					inside.push_back(i);
				else
					outside.push_back(i);
			}
			if (inside.size() != 0 && outside.size() != 0)
			{
				int i = rand() % inside.size(), j = rand() % outside.size();
				delete_from_vertex_cover(inside[i], step);
				insert_to_vertex_cover(outside[j]);
			}
		}
		int num_C = get_vc_size(C);
		if (num_C + (int)L.size() < ub)
		{
			ub = num_C + (int)L.size();
			memcpy(C_star, C, n);
			greedy_mvc_solver(C_star, L);
			remove_delta(Delta, step);
		}
	}
}

int main(int argc, char** argv)
{
	srand((unsigned int)time(0));
#ifdef DEBUG
	printf("%s\n", argv[1]);
	FILE* fin = freopen(argv[1], "r", stdin);
	if (fin == nullptr) {
		printf("Open file error\n");
		return 0;
	}
	clock_t clock_start = clock(), clock_end;
#endif
	while (scanf("%d%d", &n, &m) != EOF)
	{
		EWLS(steps * sqrt(n));
		printf("%d\n", n - get_vc_size(C_star));
		for (int i = 0; i < n; i++)
		{
			if (!C_star[i])
				printf("%d ", i + 1);
		}
		putchar('\n');
#ifdef DEBUG
		//check
		vector<int> ans;
		for (int i = 0; i < n; i++)
		{
			if (!C_star[i])
				ans.push_back(i);
		}
		for (int i = 0; i < ans.size(); i++)
		{
			for (int j = i + 1; j < ans.size(); j++)
			{
				if (w[ans[i]][ans[j]] != 0)
					printf("%d %d error\n", ans[i], ans[j]);
			}
		}
		printf("Validity check passed\n");
		clock_end = clock();
		printf("Time elapsed: %.3f s\n", (double) (clock_end - clock_start) / CLOCKS_PER_SEC);
		clock_start = clock_end;
#endif
	}
	return 0;
}

