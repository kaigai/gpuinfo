#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static int usage(int argc, char * const argv[])
{
	printf("usage: %s -s <size>\n", argv[0]);

	exit(0);
}

int main(int argc, char * const argv[])
{
	size_t		size = (1UL << 30);	/* 1GB */
	char	   *buffer;
	int			c;

	while ((c = getopt(argc, argv, "s:")) != -1)
	{
		if (c == 's')
		{
			int		i, unit;

			for (i=0; optarg[i] >= '0' && optarg[i] <= '9'; i++);

			if (strcasecmp(optarg + i, "") == 0)
				size = atol(optarg);
			else if (strcasecmp(optarg + i, "k") == 0)
				size = atol(optarg) << 10;
			else if (strcasecmp(optarg + i, "m") == 0)
				size = atol(optarg) << 20;
			else if (strcasecmp(optarg + i, "g") == 0)
				size = atol(optarg) << 30;
			else
				usage(argc, argv);
		}
		else
			usage(argc, argv);
	}

	/* memory allocation */
	buffer = malloc(size);
	if (!buffer)
	{
		printf("failed to allocate %lu bytes : %s\n", size, strerror(errno));
		return 1;
	}
	memset(buffer, 0, size);

	/* memory pinning */
	if (mlockall(MCL_FUTURE) != 0)
	{
		printf("failed to lock memory : %s\n", strerror(errno));
		return 1;
	}
	printf("OK, %s allocated and pinned %lu bytes\n", argv[0], size);

	/* infinite sleep */
	for (;;)
		sleep(60);

	return 0;
}
